import copy
import time
import numpy as np
import torch
import torch.nn.functional as F
from flcore.clients.clientbase import Client, resolve_sgd

class clientVLS(Client):

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.lamda = args.lamda
        self.teacher_model = None
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, **resolve_sgd(args, default_momentum=0.9))
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=args.learning_rate_decay_gamma)
        from utils.data_utils import read_client_data
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        counts = torch.zeros(self.num_classes, device=self.device)
        for (_, y) in train_data:
            counts[y.item()] += 1
        self.num_per_class = counts
        self.prior = counts / counts.sum()
        self.no_exist_label = torch.where(self.prior == 0)[0]
        self.exist_label = torch.where(self.prior != 0)[0]
        self.exist_prior = self.prior[self.exist_label]

    def set_parameters(self, model):
        self.teacher_model = copy.deepcopy(model)
        super().set_parameters(model)

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        start_time = time.time()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max(2, max_local_epochs // 2 + 1))
        loss_sums = {'total': 0.0, 'ce': 0.0}
        total_samples = 0
        for epoch in range(max_local_epochs):
            for (x, y) in trainloader:
                (x, y) = self._prepare_batch(x, y)
                y = y.long()
                self.optimizer.zero_grad()
                output = self.model(x)
                prior_ce = self._prior_ce_loss(output, y)
                lade = self._lade_loss(output, y)
                with torch.no_grad():
                    teach_output = self.teacher_model(x)
                vls = self._vls_loss(output, teach_output)
                loss = prior_ce + lade + vls * self.lamda
                loss.backward()
                self.optimizer.step()
                bs = y.shape[0]
                total_samples += bs
                loss_sums['total'] += loss.item() * bs
                loss_sums['ce'] += prior_ce.item() * bs
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        elapsed = time.time() - start_time
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += elapsed
        return {'local_epochs': max_local_epochs, 'train_time': elapsed, 'train_samples': self.train_samples, 'test_samples': self.test_samples, 'loss_breakdown': self._average_loss_sums(loss_sums, total_samples)}

    def _prior_ce_loss(self, output, y):
        logits = output + torch.log(self.prior + 1e-09)
        return self.loss(logits, y)

    def _lade_loss(self, y_pred, target):
        cls_weight = self.num_per_class / self.num_per_class.sum()
        balanced_prior = torch.tensor(1.0 / self.num_classes, device=self.device)
        pred_spread = (y_pred - torch.log(self.prior + 1e-09) + torch.log(balanced_prior + 1e-09)).T * (target != torch.arange(0, self.num_classes, device=self.device).view(-1, 1))
        N = pred_spread.size(-1)
        second_term = torch.logsumexp(pred_spread, -1) - np.log(N)
        loss = -torch.sum(-second_term * cls_weight)
        return loss * 0.005

    def _vls_loss(self, output, teach_output):
        if len(self.no_exist_label) == 0:
            return torch.tensor(0.0, device=self.device)
        out_log = F.log_softmax(output[:, self.no_exist_label], dim=None)
        tch_soft = F.softmax(teach_output[:, self.no_exist_label], dim=None)
        return F.kl_div(out_log, tch_soft, reduction='batchmean')

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
