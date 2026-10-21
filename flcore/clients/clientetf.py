import time
import numpy as np
import torch
import torch.nn.functional as F
from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data

def balanced_softmax_loss(labels, logits, sample_per_class, reduction='mean'):
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    return F.cross_entropy(input=logits, target=labels, reduction=reduction)

class clientETF(Client):

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.sample_per_class = torch.zeros(self.num_classes)
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        for (_, y) in train_data:
            self.sample_per_class[y.item()] += 1
        self.sample_per_class = torch.where(self.sample_per_class > 0, self.sample_per_class, torch.ones_like(self.sample_per_class))
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.0005)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=args.learning_rate_decay_gamma)
        self.personalization_rounds = getattr(args, 'etf_personalization_rounds', 20)
        self.personalization_inner_steps = getattr(args, 'etf_personalization_inner_steps', 3)
        self.etf_proto_lr = getattr(args, 'etf_proto_lr', 0.1)
        self.etf_proj_lr = getattr(args, 'etf_proj_lr', 0.05)

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
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = balanced_softmax_loss(y, output, self.sample_per_class)
                loss.backward()
                self.optimizer.step()
                bs = y.shape[0]
                total_samples += bs
                loss_sums['total'] += loss.item() * bs
                loss_sums['ce'] += loss.item() * bs
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        elapsed = time.time() - start_time
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += elapsed
        return {'local_epochs': max_local_epochs, 'train_time': elapsed, 'train_samples': self.train_samples, 'test_samples': self.test_samples, 'loss_breakdown': self._average_loss_sums(loss_sums, total_samples)}

    def personalize(self):
        trainloader = self.load_train_data()
        self.model.train()
        for epoch in range(self.local_epochs):
            for (x, y) in trainloader:
                (x, y) = self._prepare_batch(x, y)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = balanced_softmax_loss(y, output, self.sample_per_class)
                loss.backward()
                self.optimizer.step()
        for _ in range(self.personalization_rounds):
            for param in self.model.parameters():
                param.requires_grad = False
            etf_opt = self.model.head.etf_proto.detach().clone().requires_grad_(True)
            proto_optimizer = torch.optim.SGD([etf_opt], lr=self.etf_proto_lr)
            self.model.train()
            for _ in range(self.personalization_inner_steps):
                for (x, y) in trainloader:
                    (x, y) = self._prepare_batch(x, y)
                    rep = self.model.base(x)
                    projected = self.model.head.linear_proto(rep)
                    norm = torch.norm(projected, p=2, dim=1, keepdim=True).clamp(min=1e-12)
                    normalized = projected / norm
                    logits = self.model.head.scaling_train * torch.matmul(normalized, etf_opt)
                    loss = F.cross_entropy(logits, y)
                    proto_optimizer.zero_grad()
                    loss.backward()
                    proto_optimizer.step()
            with torch.no_grad():
                self.model.head.etf_proto.copy_(etf_opt.detach())
            for (name, param) in self.model.named_parameters():
                param.requires_grad = 'linear_proto' in name
            proj_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.etf_proj_lr, momentum=0.9, weight_decay=0.0005)
            self.model.train()
            for _ in range(self.personalization_inner_steps):
                for (x, y) in trainloader:
                    (x, y) = self._prepare_batch(x, y)
                    output = self.model(x)
                    loss = F.cross_entropy(output, y)
                    proj_optimizer.zero_grad()
                    loss.backward()
                    proj_optimizer.step()
        for param in self.model.parameters():
            param.requires_grad = True

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
