import time
import numpy as np
import torch
from flcore.clients.clientbase import Client, resolve_sgd

class clientRep(Client):

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.optimizer = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate, **resolve_sgd(args))
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=args.learning_rate_decay_gamma)
        self.optimizer_per = torch.optim.SGD(self.model.head.parameters(), lr=self.learning_rate, **resolve_sgd(args))
        self.learning_rate_scheduler_per = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_per, gamma=args.learning_rate_decay_gamma)
        self.plocal_epochs = args.plocal_epochs

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()
        self.model.train()
        loss_sums = {'total': 0.0, 'ce': 0.0}
        total_samples = 0
        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True
        for epoch in range(self.plocal_epochs):
            for (x, y) in trainloader:
                (x, y) = self._prepare_batch(x, y)
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer_per.zero_grad()
                loss.backward()
                self.optimizer_per.step()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max(2, max_local_epochs // 2 + 1))
        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = False
        for epoch in range(max_local_epochs):
            for (x, y) in trainloader:
                (x, y) = self._prepare_batch(x, y)
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_size = y.shape[0]
                total_samples += batch_size
                loss_value = loss.item() * batch_size
                loss_sums['total'] += loss_value
                loss_sums['ce'] += loss_value
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_per.step()
        elapsed_time = time.time() - start_time
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += elapsed_time
        return {'local_epochs': max_local_epochs, 'train_time': elapsed_time, 'train_samples': self.train_samples, 'test_samples': self.test_samples, 'loss_breakdown': self._average_loss_sums(loss_sums, total_samples)}

    def set_parameters(self, base):
        for (new_param, old_param) in zip(base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()
        for (new_buf, old_buf) in zip(base.buffers(), self.model.base.buffers()):
            if new_buf.shape == old_buf.shape:
                old_buf.data = new_buf.data.clone()

    def _prepare_batch(self, x, y):
        if isinstance(x, list):
            x[0] = x[0].to(self.device)
        else:
            x = x.to(self.device)
        y = y.to(self.device)
        return (x, y)

    def _average_loss_sums(self, loss_sums, total_samples):
        if total_samples <= 0:
            return {key: 0.0 for key in loss_sums}
        return {key: value / total_samples for (key, value) in loss_sums.items()}
