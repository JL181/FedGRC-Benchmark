import copy
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.clients.clientbase import Client, resolve_sgd

def build_dr_etf_weight(seed_weight):
    param_data = seed_weight.detach().cpu()
    class_num = param_data.shape[0]
    feature_dim = param_data.shape[1]
    assert feature_dim >= class_num, f'FedDrPlus ETF needs feature_dim >= num_classes, got d={feature_dim}, C={class_num}.'
    identity_matrix = torch.eye(class_num, dtype=torch.float64)
    one_vector = torch.ones(class_num, dtype=torch.float64)
    A = identity_matrix - 1 / class_num * (one_vector.view(-1, 1) @ one_vector.view(1, -1))
    (B, _) = np.linalg.qr(copy.deepcopy(param_data.T).double().numpy())
    B = torch.from_numpy(B)
    assert torch.allclose(torch.matmul(B.T, B), torch.eye(class_num, dtype=torch.float64), atol=1e-06), torch.max(torch.abs(torch.matmul(B.T, B) - torch.eye(class_num, dtype=torch.float64)))
    C = math.sqrt(class_num / (class_num - 1))
    etf = C * (B @ A)
    return copy.deepcopy(etf.T.detach()).float()

class DotRegressionETFHead(nn.Module):

    def __init__(self, in_features, num_classes, norm=1.0):
        super().__init__()
        self.classifier = nn.Linear(in_features, num_classes, bias=False)
        self.norm = norm
        self.in_features = in_features
        self.out_features = num_classes

    def forward(self, x):
        normalized_feature = F.normalize(x, p=2, dim=1)
        logits = self.classifier(normalized_feature * self.norm)
        return logits

class clientDrPlus(Client):

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.beta = float(getattr(args, 'drp_beta', 0.9))
        body_params = [p for (name, p) in self.model.named_parameters() if 'classifier' not in name]
        head_params = [p for (name, p) in self.model.named_parameters() if 'classifier' in name]
        selected_names = {name for (name, _) in self.model.named_parameters() if 'classifier' in name}
        expected_names = {f'head.{name}' for (name, _) in self.model.head.named_parameters()}
        if selected_names != expected_names:
            raise NotImplementedError("FedDrPlus requires a backbone without 'classifier'-named parameters so only the ETF head is frozen; use ResNet18.")
        self.optimizer = torch.optim.SGD([{'params': body_params, 'lr': self.learning_rate}, {'params': head_params, 'lr': 0.0}], **resolve_sgd(args, default_momentum=0.9, default_weight_decay=1e-05))
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=args.learning_rate_decay_gamma)
        self.MSE = nn.MSELoss()
        self.teacher = None

    def train(self):
        trainloader = self.load_train_data()
        self.teacher = copy.deepcopy(self.model)
        self.teacher.to(self.device)
        self.teacher.eval()
        for params in self.teacher.parameters():
            params.requires_grad = False
        self.model.train()
        self.model.to(self.device)
        start_time = time.time()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max(2, max_local_epochs // 2 + 1))
        loss_sums = {'total': 0.0, 'dr': 0.0, 'kd': 0.0}
        total_samples = 0
        etf_label = self.model.head.classifier.weight.detach()
        for epoch in range(max_local_epochs):
            for (x, y) in trainloader:
                (x, y) = self._prepare_batch(x, y)
                etf_labels = etf_label[y].to(self.device)
                dg_original_features = self.teacher.base(x)
                self.model.zero_grad()
                features = F.normalize(self.model.base(x), p=2, dim=1)
                original_features = self.model.base(x)
                inner_products = torch.sum(features * etf_labels, dim=1)
                dr_term = self.MSE(inner_products, torch.ones(len(y)).to(self.device))
                kd_term = self.MSE(original_features, dg_original_features)
                loss = torch.mean(self.beta * (1 / 2) * dr_term + (1.0 - self.beta) * kd_term)
                loss.backward()
                self.optimizer.step()
                batch_size = y.shape[0]
                total_samples += batch_size
                loss_sums['total'] += loss.item() * batch_size
                loss_sums['dr'] += dr_term.item() * batch_size
                loss_sums['kd'] += kd_term.item() * batch_size
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        self.teacher = None
        elapsed_time = time.time() - start_time
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += elapsed_time
        return {'local_epochs': max_local_epochs, 'train_time': elapsed_time, 'train_samples': self.train_samples, 'test_samples': self.test_samples, 'loss_breakdown': self._average_loss_sums(loss_sums, total_samples)}

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
