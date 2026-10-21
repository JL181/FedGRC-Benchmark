import time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from flcore.clients.clientbase import Client, resolve_sgd
from flcore.trainmodel.pfedgm_head import PGMFusionHead
from utils.data_utils import read_client_data

class clientPFedGM(Client):

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, **resolve_sgd(args, default_momentum=0.5, default_weight_decay=0.0005))
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=args.learning_rate_decay_gamma)
        self.feat_dim = self.model.head.in_features
        self.sample_per_class = torch.zeros(self.num_classes)
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        for (_, y) in train_data:
            self.sample_per_class[int(y.item())] += 1
        self.lam = float(getattr(args, 'pfedgm_lambda', 1.0))
        self.cov_mode = getattr(args, 'pfedgm_cov_mode', 'grad')
        self.variant = getattr(args, 'pfedgm_absent_proto', 'drop')
        self.proto_ema = float(getattr(args, 'pfedgm_proto_ema', 0.9))
        self.use_aug = int(getattr(args, 'pfedgm_aug', 0))
        self.p_epochs = int(getattr(args, 'pfedgm_local_epochs', 5))
        self.p_lr = float(getattr(args, 'pfedgm_p_lr', 0.05))
        self.lbfgs_iter = int(getattr(args, 'pfedgm_lbfgs_iter', 10))
        self.lbfgs_rounds = int(getattr(args, 'pfedgm_lbfgs_rounds', 5))
        self.lbfgs_lr = float(getattr(args, 'pfedgm_lbfgs_lr', 0.05))
        self.proto = torch.zeros(self.num_classes, self.feat_dim, device=self.device)
        self.proto_seen = torch.zeros(self.num_classes, dtype=torch.bool, device=self.device)
        self.img_size = getattr(args, 'img_size', 32)
        self.weak_aug = T.Compose([T.RandomCrop(self.img_size, padding=4), T.RandomHorizontalFlip()])

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()
        self._refresh_prototypes()
        self.model.train()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max(2, max_local_epochs // 2 + 1))
        d = self.feat_dim
        loss_sums = {'total': 0.0, 'ce': 0.0, 'rc': 0.0, 'cov': 0.0}
        total = 0
        for _ in range(max_local_epochs):
            for (x, y) in trainloader:
                (x, y) = self._prepare_batch(x, y)
                if self.use_aug:
                    x = self.weak_aug(x)
                bs = y.shape[0]
                z = self.model.base(x)
                nav = self.model.head(z)
                loss_ce = F.cross_entropy(nav, y)
                loss_rc = ((z - self.proto[y]) ** 2).sum(1).mean() / d
                loss = loss_ce + self.lam * loss_rc
                loss_cov = torch.zeros((), device=self.device)
                if self.cov_mode == 'grad':
                    cov = self.model.head.cov_logits(z.detach())
                    loss_cov = F.cross_entropy(cov, y)
                    loss = loss + loss_cov
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    self._ema_update_prototypes(z.detach(), y)
                total += bs
                loss_sums['total'] += float(loss.item()) * bs
                loss_sums['ce'] += float(loss_ce.item()) * bs
                loss_sums['rc'] += float(loss_rc.item()) * bs
                loss_sums['cov'] += float(loss_cov.item()) * bs
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        elapsed = time.time() - start_time
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += elapsed
        return {'local_epochs': max_local_epochs, 'train_time': elapsed, 'train_samples': self.train_samples, 'test_samples': self.test_samples, 'loss_breakdown': self._average_loss_sums(loss_sums, total)}

    @torch.no_grad()
    def _refresh_prototypes(self):
        self.model.eval()
        data = read_client_data(self.dataset, self.id, is_train=True)
        loader = DataLoader(data, 256, drop_last=False, shuffle=False)
        fsum = torch.zeros(self.num_classes, self.feat_dim, device=self.device)
        cnt = torch.zeros(self.num_classes, device=self.device)
        for (x, y) in loader:
            (x, y) = self._prepare_batch(x, y)
            z = self.model.base(x)
            fsum.index_add_(0, y, z)
            cnt.index_add_(0, y, torch.ones_like(y, dtype=z.dtype))
        seen = cnt > 0
        self.proto = torch.zeros_like(fsum)
        self.proto[seen] = fsum[seen] / cnt[seen].unsqueeze(1)
        self.proto_seen = seen
        self.model.train()

    @torch.no_grad()
    def _ema_update_prototypes(self, z, y):
        (C, d) = (self.num_classes, self.feat_dim)
        bsum = torch.zeros(C, d, device=self.device)
        bsum.index_add_(0, y, z)
        bcnt = torch.zeros(C, device=self.device)
        bcnt.index_add_(0, y, torch.ones_like(y, dtype=z.dtype))
        seen_b = bcnt > 0
        bmean = torch.zeros_like(bsum)
        bmean[seen_b] = bsum[seen_b] / bcnt[seen_b].unsqueeze(1)
        ema = seen_b & self.proto_seen
        new = seen_b & ~self.proto_seen
        self.proto[ema] = self.proto_ema * self.proto[ema] + (1 - self.proto_ema) * bmean[ema]
        self.proto[new] = bmean[new]
        self.proto_seen = self.proto_seen | seen_b

    def personalize(self):
        (Z, Y) = self._cache_features()
        (C, d) = (self.num_classes, self.feat_dim)
        upsilon = torch.zeros(C, d, device=self.device)
        present = torch.zeros(C, dtype=torch.bool, device=self.device)
        for c in range(C):
            m = Y == c
            if bool(m.any()):
                upsilon[c] = Z[m].mean(0)
                present[c] = True
        head = self.model.head
        mu_star = head.mu.detach().clone()
        b_star = head.b.detach().clone()
        if self.cov_mode == 'identity':
            A_star = torch.ones(C, d, device=self.device)
        else:
            A_star = head.log_A.detach().exp()
        fus = PGMFusionHead(d, C, self.lam, self.variant).to(self.device)
        fus.set_globals(mu_star, b_star, A_star, upsilon, present)
        params = [fus.mu_off, fus.b_off, fus.log_Ag, fus.log_Ac]
        opt = torch.optim.SGD(params, lr=self.p_lr, momentum=0.5, weight_decay=0.0005)
        N = Z.shape[0]
        bs = max(1, self.batch_size)
        absent = ~present
        for _ in range(self.p_epochs):
            perm = torch.randperm(N, device=self.device)
            for i in range(0, N, bs):
                idx = perm[i:i + bs]
                loss = F.cross_entropy(fus(Z[idx]), Y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                if self.variant == 'exclude':
                    with torch.no_grad():
                        fus.mu_off[absent] = 0.0
                        fus.b_off[absent] = 0.0
        for p in params:
            p.requires_grad_(False)
        fus.b_off.requires_grad_(True)
        lbfgs = torch.optim.LBFGS([fus.b_off], lr=self.lbfgs_lr, max_iter=self.lbfgs_iter, line_search_fn='strong_wolfe')
        for _ in range(self.lbfgs_rounds):

            def closure():
                lbfgs.zero_grad()
                loss = F.cross_entropy(fus(Z), Y)
                loss.backward()
                if self.variant == 'exclude':
                    fus.b_off.grad[absent] = 0.0
                return loss
            lbfgs.step(closure)
        with torch.no_grad():
            if self.variant == 'exclude':
                fus.b_off[absent] = 0.0
        self.model.head = fus
        self.model.eval()

    @torch.no_grad()
    def _cache_features(self):
        self.model.eval()
        data = read_client_data(self.dataset, self.id, is_train=True)
        loader = DataLoader(data, 256, drop_last=False, shuffle=False)
        (Z, Y) = ([], [])
        for (x, y) in loader:
            (x, y) = self._prepare_batch(x, y)
            Z.append(self.model.base(x))
            Y.append(y)
        return (torch.cat(Z), torch.cat(Y))

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
