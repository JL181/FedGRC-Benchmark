import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from flcore.clients.clientbase import Client, resolve_sgd
from utils.data_utils import read_client_data

def balanced_softmax_loss(labels, logits, sample_per_class, reduction='mean'):
    spc = (sample_per_class + 1).type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    return F.cross_entropy(input=logits, target=labels, reduction=reduction)

class clientTHE(Client):

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        rep_len = self.model.head.in_features
        self.p_head = nn.Linear(rep_len, self.num_classes, bias=False).to(self.device)
        self.opt_p_head = torch.optim.SGD(self.p_head.parameters(), lr=getattr(args, 'the_personal_lr', 0.01), **resolve_sgd(args))
        self.personal_epochs = getattr(args, 'the_personal_epochs', 1)
        self.sample_per_class = torch.zeros(self.num_classes)
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        for (_, y) in train_data:
            self.sample_per_class[y.item()] += 1
        self.local_rep = torch.zeros(rep_len, device=self.device)
        self.global_rep = torch.zeros(rep_len, device=self.device)
        self.the_steps = getattr(args, 'the_steps', 20)
        self.the_alpha = getattr(args, 'the_alpha', 0.1)
        self.the_beta = getattr(args, 'the_beta', 0.1)

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()
        self.model.train()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max(2, max_local_epochs // 2 + 1))
        loss_sums = {'total': 0.0, 'ce': 0.0}
        total_samples = 0
        for epoch in range(max_local_epochs):
            for (x, y) in trainloader:
                (x, y) = self._prepare_batch(x, y)
                rep = self.model.base(x)
                out_g = self.model.head(rep)
                loss = balanced_softmax_loss(y, out_g, self.sample_per_class)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                bs = y.shape[0]
                total_samples += bs
                loss_sums['total'] += loss.item() * bs
                loss_sums['ce'] += loss.item() * bs
        for param in self.model.base.parameters():
            param.requires_grad = False
        self.model.base.eval()
        self.p_head.train()
        all_reps_for_proto = []
        for epoch in range(self.personal_epochs):
            for (x, y) in trainloader:
                (x, y) = self._prepare_batch(x, y)
                with torch.no_grad():
                    rep = self.model.base(x)
                if epoch == self.personal_epochs - 1:
                    all_reps_for_proto.append(rep.detach())
                out_p = self.p_head(rep)
                loss_p = F.cross_entropy(out_p, y)
                self.opt_p_head.zero_grad()
                loss_p.backward()
                self.opt_p_head.step()
        if all_reps_for_proto:
            self.local_rep = torch.cat(all_reps_for_proto, dim=0).mean(dim=0)
        for param in self.model.base.parameters():
            param.requires_grad = True
        self.model.base.train()
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        elapsed = time.time() - start_time
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += elapsed
        return {'local_epochs': max_local_epochs, 'train_time': elapsed, 'train_samples': self.train_samples, 'test_samples': self.test_samples, 'loss_breakdown': self._average_loss_sums(loss_sums, total_samples)}

    def load_test_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)

    def _ensemble_predict(self, testloader):
        self.model.eval()
        self.p_head.eval()
        test_history = None
        all_preds = []
        all_labels = []
        for (x, y) in testloader:
            (x, y) = self._prepare_batch(x, y)
            B = x.shape[0]
            with torch.no_grad():
                rep = self.model.base(x)
                g_out = self.model.head(rep)
                p_out = self.p_head(rep)
            per_sample_history = []
            for i in range(B):
                if test_history is None:
                    test_history = rep[i].clone()
                per_sample_history.append(test_history.clone())
                test_history = self.the_alpha * rep[i] + (1 - self.the_alpha) * test_history
            per_sample_history = torch.stack(per_sample_history)
            test_rep = self.the_beta * rep + (1 - self.the_beta) * per_sample_history
            agg_weight = nn.Parameter(torch.ones(B, 2, device=self.device))
            ensemble_opt = torch.optim.Adam([agg_weight], lr=self.opt_p_head.defaults['lr'] * 10)
            p_feat_al = torch.norm(test_rep - self.local_rep.unsqueeze(0), dim=1).detach()
            g_feat_al = torch.norm(test_rep - self.global_rep.unsqueeze(0), dim=1).detach()
            sim = F.cosine_similarity(F.softmax(g_out, dim=1), F.softmax(p_out, dim=1), dim=1)
            for step in range(self.the_steps):
                w = F.softmax(agg_weight, dim=1)
                agg_out = w[:, 0:1] * g_out + w[:, 1:2] * p_out
                neg_entropy = (F.softmax(agg_out, dim=1) * F.log_softmax(agg_out, dim=1)).sum(dim=1)
                loss = (-sim * neg_entropy + (1 - sim) * (w[:, 0] * g_feat_al + w[:, 1] * p_feat_al)).mean()
                ensemble_opt.zero_grad()
                loss.backward()
                if agg_weight.grad is not None and torch.norm(agg_weight.grad) < 1e-05:
                    break
                ensemble_opt.step()
            with torch.no_grad():
                w = F.softmax(agg_weight, dim=1)
                agg_out = w[:, 0:1] * g_out + w[:, 1:2] * p_out
                pred = torch.argmax(agg_out, dim=1)
            all_preds.append(pred.cpu())
            all_labels.append(y.cpu())
        return (torch.cat(all_preds), torch.cat(all_labels))

    def detailed_personalized_test_metrics(self, testloader=None):
        testloaderfull = self.load_test_data() if testloader is None else testloader
        (preds, labels) = self._ensemble_predict(testloaderfull)
        correct = 0
        total = 0
        per_class_correct = [0] * self.num_classes
        per_class_total = [0] * self.num_classes
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for (pred_val, label_val) in zip(preds.tolist(), labels.tolist()):
            total += 1
            is_correct = int(pred_val == label_val)
            correct += is_correct
            per_class_total[label_val] += 1
            per_class_correct[label_val] += is_correct
            confusion_matrix[label_val, pred_val] += 1
        return {'correct': correct, 'total': total, 'per_class_correct': per_class_correct, 'per_class_total': per_class_total, 'confusion_matrix': confusion_matrix}

    def detailed_generalization_test_metrics(self, testloader):
        self.model.eval()
        correct = 0
        total = 0
        per_class_correct = [0] * self.num_classes
        per_class_total = [0] * self.num_classes
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        with torch.no_grad():
            for (x, y) in testloader:
                (x, y) = self._prepare_batch(x, y)
                rep = self.model.base(x)
                output = self.model.head(rep)
                pred = torch.argmax(output, dim=1)
                matches = pred.eq(y)
                correct += matches.sum().item()
                total += y.shape[0]
                for (lbl, plbl, ok) in zip(y.cpu().tolist(), pred.cpu().tolist(), matches.cpu().tolist()):
                    per_class_total[lbl] += 1
                    per_class_correct[lbl] += int(ok)
                    confusion_matrix[lbl, plbl] += 1
        return {'correct': correct, 'total': total, 'per_class_correct': per_class_correct, 'per_class_total': per_class_total, 'confusion_matrix': confusion_matrix}

    def test_metrics(self, model=None):
        testloader = self.load_test_data()
        (preds, labels) = self._ensemble_predict(testloader)
        test_acc = (preds == labels).sum().item()
        test_num = len(labels)
        return (test_acc, test_num, 0.0)

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()
        self.p_head.eval()
        train_num = 0
        losses = 0
        with torch.no_grad():
            for (x, y) in trainloader:
                (x, y) = self._prepare_batch(x, y)
                rep = self.model.base(x)
                out_g = self.model.head(rep)
                out_p = self.p_head(rep)
                output = out_g.detach() + out_p
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        return (losses, train_num)

    def _prepare_batch(self, x, y):
        if type(x) == type([]):
            x[0] = x[0].to(self.device)
        else:
            x = x.to(self.device)
        y = y.to(self.device)
        return (x, y)

    def _average_loss_sums(self, loss_sums, total_samples):
        if total_samples <= 0:
            return {key: 0.0 for key in loss_sums}
        return {key: value / total_samples for (key, value) in loss_sums.items()}
