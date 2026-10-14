import time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from flcore.clients.clientbase import Client
from utils.asd_aug import ASDAug
from utils.data_utils import read_client_data

class clientYoYo(Client):

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.sample_per_class = torch.zeros(self.num_classes)
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        for (_, y) in train_data:
            self.sample_per_class[y.item()] += 1
        total = self.sample_per_class.sum()
        self.local_dist = self.sample_per_class / total if total > 0 else torch.ones(self.num_classes) / self.num_classes
        self.local_eff_weights = torch.ones(self.num_classes)
        self.global_eff_weights = torch.ones(self.num_classes)
        self.warmup = getattr(args, 'yoyo_warmup', 50)
        self.gamma = getattr(args, 'yoyo_gamma', 0.1)
        self.tau = getattr(args, 'yoyo_tau', 1.5)
        self.kd_weight = getattr(args, 'yoyo_kd_weight', 1.5)
        self.kd_temp = getattr(args, 'yoyo_kd_temp', 1.5)
        self.current_round = 0
        self.img_size = getattr(args, 'img_size', 32)
        self.asd_aug_mode = getattr(args, 'asd_aug', 'trivial')
        if self.asd_aug_mode == 'minimal':
            self.weak_aug = T.Compose([T.RandomCrop(self.img_size, padding=4), T.RandomHorizontalFlip()])
            self.strong_aug = T.Compose([T.RandomCrop(self.img_size, padding=4), T.RandomHorizontalFlip(), T.RandomErasing(p=0.5, scale=(0.02, 0.33))])
        else:
            self._asd_aug = ASDAug(self.dataset, self.img_size)
            self.weak_aug = self._asd_aug.weak_view
            self.strong_aug = self._asd_aug.strong_view

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()
        self.model.train()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max(2, max_local_epochs // 2 + 1))
        loss_sums = {'total': 0.0, 'ce': 0.0, 'kd': 0.0}
        total_samples = 0
        eff_global = self.global_eff_weights.to(self.device)
        prior = (eff_global / eff_global.sum().clamp(min=1e-12)).detach()
        local_dist = self.local_dist.to(self.device)
        warmup_w = min((self.current_round + 1) / max(self.warmup, 1), 1.0)
        for epoch in range(max_local_epochs):
            for (x, y) in trainloader:
                (x, y) = self._prepare_batch(x, y)
                bs = y.shape[0]
                x_weak = self.weak_aug(x)
                x_strong = self.strong_aug(x)
                x_all = torch.cat([x_weak, x_strong], dim=0)
                y_all = torch.cat([y, y], dim=0)
                logits_all = self.model(x_all)
                prior = (1 - self.gamma) * prior + self.gamma * local_dist
                log_prior = (prior.clamp(min=1e-12) ** self.tau).log()
                logits_adj = logits_all + log_prior.unsqueeze(0)
                loss_ce = F.cross_entropy(logits_adj, y_all)
                teacher_logits = logits_adj[:bs]
                student_logits = logits_adj[bs:]
                loss_kd = torch.tensor(0.0, device=self.device)
                if warmup_w > 0:
                    with torch.no_grad():
                        teacher_pred = teacher_logits.argmax(dim=1)
                    mask = teacher_pred == y
                    if mask.any():
                        t_scaled = teacher_logits[mask].detach() / self.kd_temp
                        s_scaled = student_logits[mask] / self.kd_temp
                        loss_kd = F.kl_div(F.log_softmax(s_scaled, dim=1), F.softmax(t_scaled, dim=1), reduction='batchmean')
                    if torch.isnan(loss_kd):
                        loss_kd = torch.tensor(0.0, device=self.device)
                loss = loss_ce + self.kd_weight * warmup_w * loss_kd
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_samples += bs
                loss_sums['total'] += loss.item() * bs
                loss_sums['ce'] += loss_ce.item() * bs
                loss_sums['kd'] += loss_kd.item() * bs
        self._compute_effective_weights(trainloader)
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        elapsed = time.time() - start_time
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += elapsed
        return {'local_epochs': max_local_epochs, 'train_time': elapsed, 'train_samples': self.train_samples, 'test_samples': self.test_samples, 'loss_breakdown': self._average_loss_sums(loss_sums, total_samples)}

    @torch.no_grad()
    def _compute_effective_weights(self, trainloader):
        self.model.eval()
        eps = 1e-08
        feat_dim = None
        class_feature_sum = None
        for (x, y) in trainloader:
            (x, y) = self._prepare_batch(x, y)
            rep = self.model.base(x).detach()
            if class_feature_sum is None:
                feat_dim = rep.shape[1]
                class_feature_sum = torch.zeros(self.num_classes, feat_dim, device=self.device)
            for i in range(y.shape[0]):
                class_feature_sum[y[i].item()] += rep[i]
        class_prototype = torch.zeros_like(class_feature_sum) if class_feature_sum is not None else torch.zeros(self.num_classes, 1, device=self.device)
        for c in range(self.num_classes):
            cnt = self.sample_per_class[c].item()
            if cnt > 0 and class_feature_sum is not None:
                class_prototype[c] = class_feature_sum[c] / cnt
        eff_all = torch.zeros(self.num_classes)
        for (x, y) in trainloader:
            (x, y) = self._prepare_batch(x, y)
            rep = self.model.base(x).detach()
            mu = class_prototype[y]
            feat_centered = rep - mu
            eff_batch = torch.zeros(self.num_classes)
            for c in torch.unique(y):
                c_idx = c.item()
                feat_c = feat_centered[y == c]
                if feat_c.shape[0] < 2:
                    eff_batch[c_idx] = 1.0
                    continue
                norms = feat_c.norm(dim=1, keepdim=True)
                norm_outer = norms @ norms.T
                norm_outer = norm_outer.clamp(min=eps)
                r_matrix = feat_c @ feat_c.T / norm_outer
                n = feat_c.shape[0]
                a = torch.ones(1, n, device=r_matrix.device) / n
                b = torch.ones(n, 1, device=r_matrix.device) / n
                c_val = (a @ r_matrix @ b).item()
                if c_val < eps:
                    c_val = eps
                eff_batch[c_idx] = 1.0 / c_val
            eff_all += eff_batch
        self.local_eff_weights = eff_all
        self.model.train()

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
