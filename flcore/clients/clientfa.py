import math
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from flcore.clients.clientbase import Client, resolve_sgd
from utils.data_utils import read_client_data


def anchor_loss(features, labels, anchors, mu=0.1):
    if features.ndim != 2 or anchors.ndim != 2 or features.shape[1] != anchors.shape[1]:
        raise ValueError('FedFA expects features [B,d] and anchors [C,d]')
    if labels.ndim != 1 or labels.shape[0] != features.shape[0] or labels.numel() == 0:
        raise ValueError('FedFA anchor loss requires nonempty [B] labels')
    counts = torch.bincount(labels, minlength=anchors.shape[0])
    residual = features - anchors.detach().index_select(0, labels)
    per_example = residual.square().sum(dim=1) / counts[labels].to(features.dtype)
    return mu * (per_example.sum() / anchors.shape[0])


@torch.no_grad()
def accumulate_batch_means_(epoch_sum, features, labels):
    for cls in labels.unique().tolist():
        epoch_sum[cls].add_(features.detach()[labels == cls].mean(dim=0))


@torch.no_grad()
def update_epoch_anchors(candidate, epoch_sum, num_batches, present_classes, momentum):
    if num_batches < 1:
        raise ValueError('FedFA cannot update anchors from zero training batches')
    if not math.isfinite(momentum) or not 0 <= momentum <= 1:
        raise ValueError('fedfa_anchor_momentum must be finite and in [0,1]')
    updated = candidate.detach().clone()
    idx = present_classes.to(device=updated.device, dtype=torch.long)
    updated[idx] = momentum * candidate[idx] + (1 - momentum) * (epoch_sum[idx] / num_batches)
    return updated


class _FedFABatchSampler(BatchSampler):

    def __iter__(self):
        previous = None
        for batch in super().__iter__():
            if previous is not None:
                if len(batch) == 1 and len(previous) > 2:
                    batch = previous[-1:] + batch
                    previous = previous[:-1]
                yield previous
            previous = batch
        if previous is not None:
            yield previous

class clientFA(Client):

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        if not hasattr(self.model, 'base') or not isinstance(getattr(self.model, 'head', None), nn.Linear):
            raise ValueError('FedFA requires BaseHeadSplit with a linear head')
        if self.model.head.out_features != self.num_classes:
            raise ValueError('FedFA head output size must equal num_classes')
        if self.learning_rate_decay:
            raise ValueError('FedFA requires learning_rate_decay=False')
        self.mu = float(getattr(args, 'fedfa_mu', 0.1))
        self.anchor_momentum = float(getattr(args, 'fedfa_anchor_momentum', 0.5))
        self.calibration_lr = float(getattr(args, 'fedfa_calibration_lr', 0.001))
        if not math.isfinite(self.mu) or self.mu < 0:
            raise ValueError('fedfa_mu must be finite and nonnegative')
        if not math.isfinite(self.calibration_lr) or self.calibration_lr < 0:
            raise ValueError('fedfa_calibration_lr must be finite and nonnegative')
        if not math.isfinite(self.anchor_momentum) or not 0 <= self.anchor_momentum <= 1:
            raise ValueError('fedfa_anchor_momentum must be finite and in [0,1]')
        if self.batch_size < 1 or self.local_epochs < 1:
            raise ValueError('FedFA requires positive batch_size and local_epochs')
        data = read_client_data(self.dataset, self.id, is_train=True)
        if not data:
            raise ValueError(f'FedFA client {self.id}: empty training split')
        labels = torch.tensor([int(y) for (_, y) in data], dtype=torch.long)
        if labels.min() < 0 or labels.max() >= self.num_classes:
            raise ValueError(f'FedFA client {self.id}: training label outside [0,num_classes)')
        self.present_classes = labels.unique().to(self.device)
        self.sgd_options = resolve_sgd(args)
        self.optimizer = None
        self.learning_rate_scheduler = None
        self.calibration_optimizer = None
        self.anchors = None
        self.upload_anchors = None

    def load_train_data(self, batch_size=None):
        size = self.batch_size if batch_size is None else batch_size
        data = read_client_data(self.dataset, self.id, is_train=True)
        sampler = _FedFABatchSampler(RandomSampler(data), batch_size=size, drop_last=False)
        return DataLoader(data, batch_sampler=sampler)

    def set_anchors(self, anchors):
        if tuple(anchors.shape) != (self.num_classes, self.model.head.in_features):
            raise ValueError('FedFA received incompatible anchor dimensions')
        if not torch.isfinite(anchors).all():
            raise FloatingPointError('FedFA received non-finite anchors')
        self.anchors = anchors.detach().to(device=self.device, dtype=self.model.head.weight.dtype).clone()
        self.upload_anchors = None

    def train(self):
        if self.anchors is None:
            raise RuntimeError('FedFA must receive global anchors before training')
        loader = self.load_train_data()
        if len(loader) == 0:
            raise ValueError(f'FedFA client {self.id}: zero training batches')
        self.model.train()
        started = time.time()
        epochs = self.local_epochs
        if self.train_slow:
            epochs = np.random.randint(1, max(2, epochs // 2 + 1))
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, **self.sgd_options)
        self.calibration_optimizer = torch.optim.Adam(self.model.head.parameters(), lr=self.calibration_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        candidate = self.anchors.clone()
        class_labels = torch.arange(self.num_classes, device=self.device)
        sums = {'total': 0.0, 'ce': 0.0}
        samples = 0
        for _ in range(epochs):
            epoch_sum = torch.zeros_like(self.anchors)
            num_batches = 0
            for (x, y) in loader:
                (x, y) = (x.to(self.device), y.to(self.device))
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                features = self.model.base(x)
                ce = self.loss(self.model.head(features), y)
                align = anchor_loss(features, y, self.anchors, self.mu)
                objective = ce + align
                if not torch.isfinite(objective):
                    raise FloatingPointError(f'FedFA client {self.id}: non-finite local objective')
                self.optimizer.zero_grad()
                objective.backward()
                self.optimizer.step()
                calibration = self.loss(self.model.head(self.anchors.detach()), class_labels)
                if not torch.isfinite(calibration):
                    raise FloatingPointError(f'FedFA client {self.id}: non-finite calibration objective')
                self.calibration_optimizer.zero_grad()
                calibration.backward()
                self.calibration_optimizer.step()
                accumulate_batch_means_(epoch_sum, features, y)
                num_batches += 1
                n = y.numel()
                samples += n
                for (key, value) in (('total', objective), ('ce', ce)):
                    sums[key] += value.detach().item() * n
            candidate = update_epoch_anchors(candidate, epoch_sum, num_batches, self.present_classes, self.anchor_momentum)
        if not torch.isfinite(candidate).all():
            raise FloatingPointError(f'FedFA client {self.id}: non-finite upload anchors')
        self.upload_anchors = candidate
        elapsed = time.time() - started
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += elapsed
        return {'local_epochs': int(epochs), 'train_time': elapsed, 'train_samples': self.train_samples, 'test_samples': self.test_samples, 'loss_breakdown': {key: value / samples for (key, value) in sums.items()}}
