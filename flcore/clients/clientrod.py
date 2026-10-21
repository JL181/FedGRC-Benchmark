import copy
import torch
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client, resolve_sgd
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data

class clientROD(Client):

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.head = copy.deepcopy(self.model.head)
        self.opt_head = torch.optim.SGD(self.head.parameters(), lr=self.learning_rate, **resolve_sgd(args))
        self.learning_rate_scheduler_head = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.opt_head, gamma=args.learning_rate_decay_gamma)
        self.sample_per_class = torch.zeros(self.num_classes)
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        for (_, y) in train_data:
            self.sample_per_class[y.item()] += 1

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()
        self.model.train()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        for epoch in range(max_local_epochs):
            for (i, (x, y)) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                out_g = self.model.head(rep)
                loss_bsm = balanced_softmax_loss(y, out_g, self.sample_per_class)
                self.optimizer.zero_grad()
                loss_bsm.backward()
                self.optimizer.step()
                out_p = self.head(rep.detach())
                loss = self.loss(out_g.detach() + out_p, y)
                self.opt_head.zero_grad()
                loss.backward()
                self.opt_head.step()
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_head.step()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def test_metrics(self, model=None):
        testloader = self.load_test_data()
        if model == None:
            model = self.model
        model.eval()
        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        with torch.no_grad():
            for (x, y) in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                out_g = self.model.head(rep)
                out_p = self.head(rep.detach())
                output = out_g.detach() + out_p
                test_acc += torch.sum(torch.argmax(output, dim=1) == y).item()
                test_num += y.shape[0]
                y_prob.append(F.softmax(output).detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        return (test_acc, test_num, auc)

    def detailed_personalized_test_metrics(self, testloader=None):
        testloaderfull = self.load_test_data() if testloader is None else testloader
        self.model.eval()
        self.head.eval()
        correct = 0
        total = 0
        per_class_correct = [0 for _ in range(self.num_classes)]
        per_class_total = [0 for _ in range(self.num_classes)]
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        with torch.no_grad():
            for (x, y) in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                out_g = self.model.head(rep)
                out_p = self.head(rep.detach())
                output = out_g.detach() + out_p
                pred = torch.argmax(output, dim=1)
                matches = pred.eq(y)
                correct += matches.sum().item()
                total += y.shape[0]
                y_cpu = y.detach().cpu().tolist()
                pred_cpu = pred.detach().cpu().tolist()
                matches_cpu = matches.detach().cpu().tolist()
                for (label, predicted_label, is_correct) in zip(y_cpu, pred_cpu, matches_cpu):
                    per_class_total[label] += 1
                    per_class_correct[label] += int(is_correct)
                    confusion_matrix[label, predicted_label] += 1
        return {'correct': correct, 'total': total, 'per_class_correct': per_class_correct, 'per_class_total': per_class_total, 'confusion_matrix': confusion_matrix}

    def detailed_generalization_test_metrics(self, testloader):
        self.model.eval()
        correct = 0
        total = 0
        per_class_correct = [0 for _ in range(self.num_classes)]
        per_class_total = [0 for _ in range(self.num_classes)]
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        with torch.no_grad():
            for (x, y) in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                output = self.model.head(rep)
                pred = torch.argmax(output, dim=1)
                matches = pred.eq(y)
                correct += matches.sum().item()
                total += y.shape[0]
                y_cpu = y.detach().cpu().tolist()
                pred_cpu = pred.detach().cpu().tolist()
                matches_cpu = matches.detach().cpu().tolist()
                for (label, predicted_label, is_correct) in zip(y_cpu, pred_cpu, matches_cpu):
                    per_class_total[label] += 1
                    per_class_correct[label] += int(is_correct)
                    confusion_matrix[label, predicted_label] += 1
        return {'correct': correct, 'total': total, 'per_class_correct': per_class_correct, 'per_class_total': per_class_total, 'confusion_matrix': confusion_matrix}

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()
        train_num = 0
        losses = 0
        with torch.no_grad():
            for (x, y) in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                out_g = self.model.head(rep)
                out_p = self.head(rep.detach())
                output = out_g.detach() + out_p
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        return (losses, train_num)

def balanced_softmax_loss(labels, logits, sample_per_class, reduction='mean'):
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss
