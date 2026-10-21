import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.correlation_tools import cov_nearest
from flcore.clients.clientbase import Client, resolve_sgd
from utils.data_utils import read_client_data
import torchmin

class clientFDA(Client):

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.eps = 0.0001
        self.cv_seed = args.seed
        self.num_cv_folds = 2
        self.min_samples = self.num_cv_folds
        self.single_beta = bool(getattr(args, 'single_beta', 1))
        self.local_beta = getattr(args, 'local_beta', False)
        if self.local_beta:
            self.single_beta = True
        D = self.model.head.in_features
        self.means = torch.rand(self.num_classes, D)
        self.covariance = torch.eye(D)
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        counts = torch.zeros(self.num_classes)
        for (_, y) in train_data:
            counts[y.item()] += 1
        priors = counts / counts.sum()
        self.priors = priors + self.eps
        self.priors = self.priors / self.priors.sum()
        self.class_counts = counts
        self.global_means = self.means.clone()
        self.global_covariance = self.covariance.clone()
        self.adaptive_means = self.means.clone()
        self.adaptive_covariance = self.covariance.clone()
        self.means_beta = torch.ones(self.num_classes) * 0.5
        self.cov_beta = torch.tensor([0.5])
        self.optimizer = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate, **resolve_sgd(args, default_momentum=0.5, default_weight_decay=0.0005))
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=args.learning_rate_decay_gamma)

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()
        self.means = self.means.to(self.device)
        self.covariance = self.covariance.to(self.device)
        self.adaptive_means = self.adaptive_means.to(self.device)
        self.adaptive_covariance = self.adaptive_covariance.to(self.device)
        self.global_means = self.global_means.to(self.device)
        self.global_covariance = self.global_covariance.to(self.device)
        self._set_lda_weights(self.global_means, self.global_covariance, self.priors)
        for p in self.model.head.parameters():
            p.requires_grad_(False)
        self.model.train()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max(2, max_local_epochs // 2 + 1))
        loss_sums = {'total': 0.0, 'ce': 0.0}
        total_samples = 0
        all_feats = []
        all_labels = []
        for epoch in range(max_local_epochs):
            for (x, y) in trainloader:
                (x, y) = self._prepare_batch(x, y)
                y = y.long()
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                all_feats.append(rep.detach())
                all_labels.append(y.cpu().numpy())
                bs = y.shape[0]
                total_samples += bs
                loss_sums['total'] += loss.item() * bs
                loss_sums['ce'] += loss.item() * bs
        feats = torch.cat(all_feats, dim=0)
        labels = np.concatenate(all_labels, axis=0)
        self._solve_beta(feats, labels)
        (means_mle, scatter, _, counts) = self._compute_mle(feats, labels)
        D = feats.shape[1]
        cov_mle = scatter / max(np.sum(counts) - 1, 1) + self.eps + torch.eye(D, device=self.device)
        cov_psd = cov_nearest(cov_mle.cpu().numpy(), method='clipped')
        cov_psd = torch.tensor(cov_psd, dtype=torch.float32, device=self.device)
        means_mle = torch.stack([means_mle[i] if means_mle[i] is not None else self.global_means[i] for i in range(self.num_classes)])
        self._update_adaptive(means_mle, cov_psd)
        for p in self.model.head.parameters():
            p.requires_grad_(True)
        self.means = self.means.cpu()
        self.covariance = self.covariance.cpu()
        self.adaptive_means = self.adaptive_means.cpu()
        self.adaptive_covariance = self.adaptive_covariance.cpu()
        self.global_means = self.global_means.cpu()
        self.global_covariance = self.global_covariance.cpu()
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        elapsed = time.time() - start_time
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += elapsed
        return {'local_epochs': max_local_epochs, 'train_time': elapsed, 'train_samples': self.train_samples, 'test_samples': self.test_samples, 'loss_breakdown': self._average_loss_sums(loss_sums, total_samples)}

    def set_fda_eval(self):
        self.means = self.means.to(self.device)
        self.covariance = self.covariance.to(self.device)
        self.adaptive_means = self.adaptive_means.to(self.device)
        self.adaptive_covariance = self.adaptive_covariance.to(self.device)
        self.global_means = self.global_means.to(self.device)
        self.global_covariance = self.global_covariance.to(self.device)
        self.model.eval()
        (feats, labels) = self._compute_feats_from_train()
        self._solve_beta(feats, labels)
        (means_mle, scatter, _, counts) = self._compute_mle(feats, labels)
        D = feats.shape[1]
        cov_mle = scatter / max(np.sum(counts) - 1, 1) + self.eps + torch.eye(D, device=self.device)
        cov_psd = cov_nearest(cov_mle.cpu().numpy(), method='clipped')
        cov_psd = torch.tensor(cov_psd, dtype=torch.float32, device=self.device)
        means_mle = torch.stack([means_mle[i] if means_mle[i] is not None else self.global_means[i] for i in range(self.num_classes)])
        self._update_adaptive(means_mle, cov_psd)
        self._set_lda_weights(self.adaptive_means, self.adaptive_covariance, self.priors)
        self.means = self.means.cpu()
        self.covariance = self.covariance.cpu()
        self.adaptive_means = self.adaptive_means.cpu()
        self.adaptive_covariance = self.adaptive_covariance.cpu()
        self.global_means = self.global_means.cpu()
        self.global_covariance = self.global_covariance.cpu()

    def _set_lda_weights(self, means, covariance, priors=None):
        if priors is None:
            priors = self.priors
        with torch.no_grad():
            means_c = means.cpu().float()
            cov_c = covariance.cpu().float()
            priors_c = priors.cpu().float()
            D = cov_c.shape[0]
            cov_reg = (1 - self.eps) * cov_c + self.eps * (torch.trace(cov_c) / D) * torch.eye(D)
            coefs = torch.linalg.lstsq(cov_reg, means_c.T).solution.T
            intercepts = -0.5 * torch.diag(means_c @ coefs.T) + torch.log(priors_c)
            self.model.head.weight.data = coefs.detach().to(self.device)
            self.model.head.bias.data = intercepts.detach().to(self.device)

    def _compute_mle(self, feats, labels):
        means = [None] * self.num_classes
        counts = np.bincount(labels.astype(int), minlength=self.num_classes)
        priors = counts / float(len(labels))
        for y in np.unique(labels):
            y = int(y)
            if counts[y] >= self.min_samples:
                means[y] = torch.mean(feats[labels == y], dim=0)
        feats_centered = []
        for y in np.unique(labels):
            y = int(y)
            mu = means[y] if means[y] is not None else self.global_means[y]
            feats_centered.append(feats[labels == y] - mu)
        feats_centered = torch.cat(feats_centered, dim=0)
        scatter = feats_centered.T @ feats_centered
        return (means, scatter, priors, counts)

    def _compute_feats_from_train(self):
        trainloader = self.load_train_data()
        all_feats = []
        all_labels = []
        with torch.no_grad():
            for (x, y) in trainloader:
                (x, y) = self._prepare_batch(x, y)
                rep = self.model.base(x)
                all_feats.append(rep)
                all_labels.append(y.cpu().numpy())
        return (torch.cat(all_feats, dim=0), np.concatenate(all_labels, axis=0))

    def _solve_beta(self, feats, labels, seed=None):
        seed = self.cv_seed if seed is None else seed
        if self.local_beta:
            self.means_beta = torch.ones(1, device=self.device)
            self.cov_beta = torch.ones(1, device=self.device)
            return
        (vals, counts) = np.unique(labels, return_counts=True)
        pruned_feats = feats.clone()
        pruned_labels = labels.copy()
        for (v, c) in zip(vals, counts):
            if c < self.num_cv_folds:
                mask = pruned_labels != v
                pruned_feats = pruned_feats[mask]
                pruned_labels = pruned_labels[mask]
        try:
            skf = StratifiedKFold(n_splits=self.num_cv_folds, random_state=seed, shuffle=True)
            (fold_means, fold_covs) = ([], [])
            (fold_feats_te, fold_labels_te) = ([], [])
            for (train_idx, test_idx) in skf.split(pruned_feats.cpu(), pruned_labels):
                (f_tr, l_tr) = (pruned_feats[train_idx], pruned_labels[train_idx])
                (f_te, l_te) = (pruned_feats[test_idx], pruned_labels[test_idx])
                (means_tr, scatter_tr, _, cts) = self._compute_mle(f_tr, l_tr)
                D = f_tr.shape[1]
                cov_tr = scatter_tr / max(np.sum(cts) - 1, 1) + self.eps + torch.eye(D, device=self.device)
                cov_psd = cov_nearest(cov_tr.cpu().numpy(), method='clipped')
                cov_psd = torch.tensor(cov_psd, dtype=torch.float32, device=self.device)
                ms = torch.stack([means_tr[i] if means_tr[i] is not None else self.global_means[i] for i in range(self.num_classes)])
                fold_means.append(ms)
                fold_covs.append(cov_psd)
                fold_feats_te.append(f_te)
                fold_labels_te.append(l_te)
        except Exception:
            self.means_beta = torch.zeros(1, device=self.device)
            self.cov_beta = torch.zeros(1, device=self.device)
            return

        def cv_loss(beta):
            total = torch.tensor(0.0, device=self.device)
            for i in range(self.num_cv_folds):
                b = beta.clamp(0, 1)
                if self.single_beta:
                    m = b * fold_means[i] + (1 - b) * self.global_means
                    c = b * fold_covs[i] + (1 - b) * self.global_covariance
                else:
                    m = b[0] * fold_means[i] + (1 - b[0]) * self.global_means
                    c = b[-1] * fold_covs[i] + (1 - b[-1]) * self.global_covariance
                logits = self._lda_classify(fold_feats_te[i], m, c, self.priors)
                total = total + F.cross_entropy(logits, torch.tensor(fold_labels_te[i], dtype=torch.long, device=self.device))
            return total
        try:
            dim = 1 if self.single_beta else 2
            x0 = 0.5 * torch.ones(dim, device=self.device)
            result = torchmin.minimize(cv_loss, x0=x0, method='l-bfgs', max_iter=10, options={'gtol': 0.001})
            x = result.x.cpu().clamp(0, 1)
            self.means_beta = torch.ones_like(self.means_beta) * x[0]
            self.cov_beta = x[-1] if not self.single_beta else x[0]
            self.means_beta = self.means_beta.to(self.device)
            self.cov_beta = self.cov_beta.to(self.device) if isinstance(self.cov_beta, torch.Tensor) else torch.tensor([self.cov_beta], device=self.device)
        except Exception:
            pass

    def _lda_classify(self, feats, means, covariance, priors):
        means_c = means.cpu().float()
        cov_c = covariance.cpu().float()
        priors_c = priors.cpu().float()
        feats_c = feats.cpu().float()
        D = cov_c.shape[0]
        cov_reg = (1 - self.eps) * cov_c + self.eps * (torch.trace(cov_c) / D) * torch.eye(D)
        coefs = torch.linalg.lstsq(cov_reg, means_c.T).solution.T
        intercepts = -0.5 * torch.diag(means_c @ coefs.T) + torch.log(priors_c)
        result = feats_c @ coefs.T + intercepts
        return result.to(self.device)

    def _update_adaptive(self, means_mle, cov_mle):
        self.means_beta = self.means_beta.to(self.device)
        self.cov_beta = self.cov_beta.to(self.device)
        self.means.data = means_mle.data
        self.covariance.data = cov_mle.data
        self.adaptive_means.data = self.means_beta.unsqueeze(1) * means_mle + (1 - self.means_beta.unsqueeze(1)) * self.global_means
        self.adaptive_covariance.data = self.cov_beta * cov_mle + (1 - self.cov_beta) * self.global_covariance
        self.means_beta = self.means_beta.cpu()
        self.cov_beta = self.cov_beta.cpu()

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
