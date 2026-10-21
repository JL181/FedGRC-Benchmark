import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data

@torch.no_grad()
def install_eta_into_head(head, eta):
    head.weight.data.copy_(eta[1:, :].t().to(head.weight.device, head.weight.dtype))
    head.bias.data.copy_(eta[0, :].to(head.bias.device, head.bias.dtype))

class clientFedLog(Client):

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.feat_dim = self.model.head.in_features
        self.m = self.feat_dim + 1
        self.m_total = self.m * self.num_classes
        self.flog_lr = float(getattr(args, 'flog_lr', 0.0005))
        self.stat_batch_size = int(getattr(args, 'flog_stat_batch_size', 100))
        for p in self.model.head.parameters():
            p.requires_grad_(False)
        self.upload_ss = None
        self.upload_n = 0

    def set_global_head(self, eta):
        if eta is None:
            return
        install_eta_into_head(self.model.head, eta)

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
            optimizer = torch.optim.Adam(self.model.base.parameters(), lr=self.flog_lr)
            for (x, y) in trainloader:
                (x, y) = self._prepare_batch(x, y)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_size = y.shape[0]
                total_samples += batch_size
                loss_value = loss.item() * batch_size
                loss_sums['total'] += loss_value
                loss_sums['ce'] += loss_value
        (self.upload_ss, self.upload_n) = self.compute_sufficient_statistics()
        elapsed_time = time.time() - start_time
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += elapsed_time
        return {'local_epochs': max_local_epochs, 'train_time': elapsed_time, 'train_samples': self.train_samples, 'test_samples': self.test_samples, 'loss_breakdown': self._average_loss_sums(loss_sums, total_samples)}

    @torch.no_grad()
    def compute_sufficient_statistics(self):
        self.model.eval()
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        loader = DataLoader(train_data, self.stat_batch_size, drop_last=False, shuffle=False)
        ss = torch.zeros(self.num_classes, self.m, device=self.device)
        n = 0
        for (x, y) in loader:
            (x, y) = self._prepare_batch(x, y)
            phi = self.model.base(x)
            f = torch.cat((torch.ones(phi.shape[0], 1, device=phi.device), phi), dim=1)
            for c in torch.unique(y):
                ss[c] += f[y == c].sum(0)
            n += y.shape[0]
        self.model.train()
        return (ss.view(-1), n)

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
