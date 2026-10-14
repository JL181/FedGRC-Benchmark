import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import beta as scipy_beta
from torch.utils.data import DataLoader
from flcore.clients.clientbase import Client, resolve_sgd
from utils.data_utils import read_client_data

class CategoryDistributionModel(nn.Module):

    def __init__(self, input_dim, q_level):
        super(CategoryDistributionModel, self).__init__()
        self.category_parameters = nn.Parameter(torch.rand(q_level, input_dim))
        self.q_level = q_level
        self.input_dim = input_dim

    def forward(self, x, log=True):
        test_outputs = torch.gather(self.category_parameters, 0, x)
        if log:
            test_outputs = torch.log(test_outputs * (1 - 0.2 * self.q_level) + 0.2)
        score = test_outputs
        return_value = score.sum(1)
        if not log:
            return_value = score.mean(1)
        return return_value.unsqueeze(1).detach()

def fsq_quantize(tensor, num_values):
    tanh_output = F.tanh(tensor)
    normalized = tanh_output * (num_values - 1)
    quantized_output = torch.round(normalized)
    return (quantized_output - normalized).detach() + normalized

class clientBTFL(Client):

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.rep_len = self.model.head.in_features
        self.q_level = int(getattr(args, 'btfl_q_level', 2))
        self.kb = float(getattr(args, 'btfl_kb', 1.0))
        self.sigma = float(getattr(args, 'btfl_sigma', 0.01))
        self.update_condition = float(getattr(args, 'btfl_update_condition', 5.0))
        self.personal_lr = float(getattr(args, 'btfl_personal_lr', 0.01))
        self.personal_epochs = int(getattr(args, 'btfl_personal_epochs', 1))
        self.ent_norm = float(getattr(args, 'btfl_ent_norm', math.log(100)))
        self.num_clients_total = int(getattr(args, 'num_clients', 1))
        self.sgd_kwargs = resolve_sgd(args)
        self.personal_head = nn.Linear(self.rep_len, self.num_classes, bias=False).to(self.device)
        self.pro = CategoryDistributionModel(self.rep_len, self.q_level).to(self.device)
        self.global_pro = CategoryDistributionModel(self.rep_len, self.q_level).to(self.device)
        self.local_rep = torch.zeros(self.rep_len, device=self.device)
        self.local_rep_sigma = torch.ones(self.rep_len, device=self.device)
        self.global_rep_by_mean = torch.zeros(self.rep_len, device=self.device)
        self.local_en = math.log(max(self.num_classes, 2))
        self.global_en = math.log(max(self.num_classes, 2))
        self.stats_ready = False
        self.num_head = 2
        self.agg_weight = torch.ones(self.batch_size, self.num_head, device=self.device) / self.num_head
        self._warned_nan_gate = False
        self._warned_singleton_batch = False
        self._reset_gate_state()

    def load_test_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)

    def load_full_train_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=False, shuffle=False)

    def _prepare_batch(self, x, y):
        if type(x) == type([]):
            x[0] = x[0].to(self.device)
        else:
            x = x.to(self.device)
        y = y.to(self.device)
        return (x, y)

    @staticmethod
    def _batch_len(x):
        return (x[0] if type(x) == type([]) else x).shape[0]

    def _skip_singleton_train_batch(self, x):
        if self._batch_len(x) > 1:
            return False
        if not self._warned_singleton_batch:
            self._warned_singleton_batch = True
            print(f'[BTFL][note] client {self.id}: skipping a 1-sample tail batch in a train-mode statistic sweep (BatchNorm needs >1 sample per channel; the official code raises ValueError here). See PORT-NOTES-btfl.md smoke fix #1.')
        return True

    def _snapshot_buffers(self):
        return {name: buf.detach().clone() for (name, buf) in self.model.named_buffers()}

    def _restore_buffers(self, snapshot):
        with torch.no_grad():
            for (name, buf) in self.model.named_buffers():
                saved = snapshot.get(name)
                if saved is not None and saved.shape == buf.shape:
                    buf.copy_(saved)

    @torch.no_grad()
    def compute_local_dle(self):
        loader = self.load_full_train_data()
        self.model.train()
        snapshot = self._snapshot_buffers()
        rows = []
        for (x, y) in loader:
            (x, _) = self._prepare_batch(x, y)
            if self._skip_singleton_train_batch(x):
                continue
            rep = self.model.base(x)
            rows.append(fsq_quantize(rep, self.q_level).detach().cpu())
        self._restore_buffers(snapshot)
        if len(rows) == 0:
            return self.pro.category_parameters.detach().cpu().clone()
        matrix = torch.cat(rows, dim=0)
        n_samples = matrix.shape[0]
        empirical_distributions = torch.stack([torch.histc(col.float(), bins=self.q_level, min=0, max=self.q_level - 1) / n_samples for col in matrix.t()], dim=1)
        self.pro.category_parameters = nn.Parameter(empirical_distributions.to(self.device))
        return empirical_distributions.clone()

    def receive_global_dle(self, global_dle):
        with torch.no_grad():
            self.global_pro.category_parameters.data = global_dle.to(self.device).clone()
            n = self.num_clients_total
            if n > 1:
                self.global_pro.category_parameters.data = self.global_pro.category_parameters.data * n / (n - 1) - self.pro.category_parameters.data / (n - 1)

    def receive_global_reps(self, stacked_reps, n_participants):
        with torch.no_grad():
            mean_rep = stacked_reps.to(self.device).mean(dim=0)
            n = self.num_clients_total
            if n > 1:
                self.global_rep_by_mean = mean_rep * n / (n - 1) - self.local_rep / (n - 1)
            else:
                self.global_rep_by_mean = mean_rep
        del n_participants

    def train(self):
        start_time = time.time()
        (p_ce_sum, p_samples) = (0.0, 0)
        (ce_sum, l1_sum, b_samples) = (0.0, 0.0, 0)
        loader_a = self.load_train_data()
        self.model.train()
        snapshot = self._snapshot_buffers()
        grad_flags = [p.requires_grad for p in self.model.parameters()]
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.personal_head.requires_grad_(True)
        self.personal_head.train()
        opt_personal = torch.optim.SGD(self.personal_head.parameters(), lr=self.personal_lr, **self.sgd_kwargs)
        for _ in range(self.personal_epochs):
            for (x, y) in loader_a:
                (x, y) = self._prepare_batch(x, y)
                y = y.long()
                with torch.no_grad():
                    rep = self.model.base(x)
                p_out = self.personal_head(rep)
                loss_p = F.cross_entropy(p_out, y)
                opt_personal.zero_grad()
                loss_p.backward()
                opt_personal.step()
                bs = y.shape[0]
                p_samples += bs
                p_ce_sum += loss_p.item() * bs
        (reps, l_ents, g_ents) = ([], [], [])
        with torch.no_grad():
            for (x, y) in self.load_full_train_data():
                (x, y) = self._prepare_batch(x, y)
                if self._skip_singleton_train_batch(x):
                    continue
                rep = self.model.base(x)
                g_out = self.model.head(rep)
                p_out = self.personal_head(rep)
                reps.append(rep)
                l_ents.append(-(p_out.softmax(1) * p_out.log_softmax(1)).sum(1))
                g_ents.append(-(g_out.softmax(1) * g_out.log_softmax(1)).sum(1))
        if len(reps) > 0:
            self.local_rep = torch.cat(reps).mean(dim=0).detach()
            self.local_rep_sigma = torch.ones_like(self.local_rep)
            self.local_en = torch.cat(l_ents).mean(dim=0).item()
            self.global_en = torch.cat(g_ents).mean(dim=0).item()
            self.stats_ready = True
        self._restore_buffers(snapshot)
        for (p, flag) in zip(self.model.parameters(), grad_flags):
            p.requires_grad_(flag)
        loader_b = self.load_train_data()
        self.model.train()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max(2, max_local_epochs // 2 + 1))
        for _ in range(max_local_epochs):
            for (x, y) in loader_b:
                (x, y) = self._prepare_batch(x, y)
                y = y.long()
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss_ce = self.loss(output, y)
                loss_l1 = self.sigma * torch.norm(rep, p=1) / 32
                loss = loss_ce + loss_l1
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                bs = y.shape[0]
                b_samples += bs
                ce_sum += loss_ce.item() * bs
                l1_sum += loss_l1.item() * bs
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        elapsed = time.time() - start_time
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += elapsed
        ce = ce_sum / b_samples if b_samples > 0 else 0.0
        l1 = l1_sum / b_samples if b_samples > 0 else 0.0
        ce_personal = p_ce_sum / p_samples if p_samples > 0 else 0.0
        return {'local_epochs': max_local_epochs, 'train_time': elapsed, 'train_samples': self.train_samples, 'test_samples': self.test_samples, 'loss_breakdown': {'total': ce + l1 + ce_personal, 'ce': ce, 'l1_rep': l1, 'ce_personal': ce_personal}}

    def _reset_gate_state(self):
        self.beta_alpha = 1.0
        self.beta_beta = 1.0
        self.bin_size = 0.05
        self.bin_count = [0] * int(1 / self.bin_size)
        self.beta_distribution = scipy_beta(self.beta_alpha, self.beta_beta)

    def _calculate_samplewise_weight(self, rep, g_out, p_out):
        self.g_entropy = -(g_out.softmax(1) * g_out.log_softmax(1)).sum(1)
        self.l_entropy = -(p_out.softmax(1) * p_out.log_softmax(1)).sum(1)
        self.g_Bol = torch.exp(-self.g_entropy / self.kb).cpu().numpy()
        self.l_Bol = torch.exp(-self.l_entropy / self.kb).cpu().numpy()
        self.div_bol = self.l_Bol / self.g_Bol
        dx = torch.ones(rep.shape[0]).numpy()
        self.div_bol = self.div_bol * dx
        self.div_entropy = (self.g_entropy / self.l_entropy).cpu().numpy() * dx
        self.abother_deal = (self.l_entropy - self.g_entropy).cpu().numpy() * dx
        features_temp = fsq_quantize(rep, self.q_level).to(torch.int64)
        features_temp = features_temp.clamp_(0, self.q_level - 1)
        self.disc_local = self.pro(features_temp).cpu().numpy().flatten()
        self.disc_global = self.global_pro(features_temp).cpu().numpy().flatten()
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            self.div_vmf = self.disc_local / self.disc_global
            self.div_vmf = torch.exp(self.pro(features_temp) - self.global_pro(features_temp)).cpu().numpy().flatten()
            delta_sl = -(self.local_en - self.l_entropy).cpu().numpy() / self.local_en
            delta_sg = -(self.ent_norm - self.g_entropy).cpu().numpy() / self.ent_norm
            self.pow_exp_local = np.exp(delta_sl)
            self.pow_exp_global = np.exp(delta_sg)
            self.dim_vmf = np.exp((self.disc_local * self.pow_exp_local - self.disc_global * self.pow_exp_global) / self.rep_len)
            self.lams_np = (self.kb * np.tanh(self.dim_vmf - 1) + 1) * (self.dim_vmf > 1) + (self.dim_vmf < 1) * (1 / (self.kb * np.tanh(1 / self.dim_vmf - 1) + 1))
            a = 0.01
            b = 0.99
            n = 100
            x = np.linspace(a, b, n)
            numerator = self.beta_distribution.pdf(x) * x
            denominator = x[None, :] + (1 - x)[None, :] * np.asarray(self.lams_np)[:, None]
            integrals = np.trapz(numerator[None, :] / denominator, x, axis=1)
        self.agg_weight[:, 0] = torch.from_numpy(integrals).to(device=self.agg_weight.device, dtype=self.agg_weight.dtype)
        self.agg_weight[:, 1] = 1 - self.agg_weight[:, 0]
        for data in self.agg_weight[:, 0].cpu().view(-1).tolist():
            if not math.isfinite(data):
                continue
            index = int(data // self.bin_size)
            if 0 <= index < len(self.bin_count):
                self.bin_count[index] += 1
        self.condition = self.div_vmf
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            cond_ratio = self.condition / (1 + self.condition)
        self.abstract_weight = self.agg_weight.detach().cpu().clone()
        self.abstract_weight[:, 1] = torch.from_numpy(np.asarray(cond_ratio)).to(self.abstract_weight.dtype)
        self.abstract_weight[:, 0] = 1 - self.abstract_weight[:, 1]
        self.abstract_entropy = -torch.sum(self.abstract_weight * torch.log(self.abstract_weight), dim=1)
        teacher_prob = F.softmax(g_out, dim=1)
        student_log_prob = F.log_softmax(p_out, dim=1)
        self.kl_g_p = F.kl_div(student_log_prob, teacher_prob, reduction='none').sum(1)
        self.kl_p_g = F.kl_div(F.log_softmax(g_out, dim=1), F.softmax(p_out, dim=1), reduction='none').sum(1)
        condition_t = torch.from_numpy(np.asarray(self.condition, dtype=np.float32))
        mask_alpha = (condition_t < 1) * (self.g_entropy.cpu() < self.global_en) * (self.l_entropy.cpu() > self.local_en)
        delta_alpha = torch.sum(mask_alpha).item()
        mask_beta = (condition_t > 1) * (self.l_entropy.cpu() < self.local_en) * (self.g_entropy.cpu() > self.global_en)
        delta_beta = torch.sum(mask_beta).item()
        self.beta_alpha = self.beta_alpha + delta_alpha
        self.beta_beta = self.beta_beta + delta_beta
        restart_flag = self.beta_alpha + self.beta_beta
        if restart_flag > self.update_condition:
            self.beta_alpha = self.beta_alpha / restart_flag * self.update_condition / 10 + 1
            self.beta_beta = self.beta_beta / restart_flag * self.update_condition / 10 + 1
        self.beta_distribution = scipy_beta(self.beta_alpha, self.beta_beta)

    def _multi_head_inference(self, g_out, p_out):
        y_g_origin = F.log_softmax(g_out, dim=1)
        y_l_origin = F.log_softmax(p_out, dim=1)
        self.l_entropy = self.l_entropy.unsqueeze(1)
        self.kl_p_g = self.kl_p_g.unsqueeze(1)
        self.g_entropy = self.g_entropy.unsqueeze(1)
        self.kl_g_p = self.kl_g_p.unsqueeze(1)
        fm_g = self.l_entropy + self.kl_p_g
        y_g = (self.l_entropy * y_g_origin + self.kl_p_g * y_l_origin) / fm_g
        fm_l = self.g_entropy + self.kl_g_p
        y_l = (self.g_entropy * y_l_origin + self.kl_g_p * y_g_origin) / fm_l
        y_g = y_g_origin * self.g_entropy / (self.l_entropy + self.g_entropy)
        y_l = y_l_origin * self.l_entropy / (self.l_entropy + self.g_entropy)
        del y_g, y_l, fm_g, fm_l
        agg_output = self.agg_weight[:, 0].unsqueeze(1) * y_g_origin + self.agg_weight[:, 1].unsqueeze(1) * y_l_origin
        return agg_output

    @torch.no_grad()
    def _btfl_predict(self, testloader):
        self.model.eval()
        self.personal_head.eval()
        self._reset_gate_state()
        (preds, labels) = ([], [])
        for (x, y) in testloader:
            (x, y) = self._prepare_batch(x, y)
            rep = self.model.base(x)
            g_out = self.model.head(rep)
            p_out = self.personal_head(rep)
            test_rep = rep.detach()
            self.agg_weight = torch.ones(y.shape[0], self.num_head, device=self.device)
            self._calculate_samplewise_weight(test_rep, g_out, p_out)
            if not self._warned_nan_gate and (not bool(torch.isfinite(self.agg_weight).all())):
                self._warned_nan_gate = True
                print(f'[BTFL][WARNING] client {self.id}: the per-sample gate is NaN/Inf (q_level={self.q_level}). With q_level > 5 the official smoothing log(p * (1 - 0.2 * q_level) + 0.2) takes the log of a non-positive number; the official code raises ValueError here. Predictions degenerate to argmax of a NaN vector. Use --btfl_q_level 2 to reproduce the official exps/*.py setting.')
            agg_output = self._multi_head_inference(g_out, p_out)
            preds.append(torch.argmax(agg_output, dim=1).cpu())
            labels.append(y.cpu())
        self.agg_weight = torch.ones(self.batch_size, self.num_head, device=self.device) / self.num_head
        if len(preds) == 0:
            return (torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
        return (torch.cat(preds), torch.cat(labels))

    def _detail_from_preds(self, preds, labels):
        correct = 0
        total = 0
        per_class_correct = [0] * self.num_classes
        per_class_total = [0] * self.num_classes
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for (pred_val, label_val) in zip(preds.tolist(), labels.tolist()):
            total += 1
            is_correct = int(pred_val == label_val)
            correct += is_correct
            per_class_total[label_val] += 1
            per_class_correct[label_val] += is_correct
            confusion_matrix[label_val, pred_val] += 1
        return {'correct': correct, 'total': total, 'per_class_correct': per_class_correct, 'per_class_total': per_class_total, 'confusion_matrix': confusion_matrix}

    def detailed_personalized_test_metrics(self, testloader=None):
        testloaderfull = self.load_test_data() if testloader is None else testloader
        (preds, labels) = self._btfl_predict(testloaderfull)
        return self._detail_from_preds(preds, labels)

    def detailed_generalization_test_metrics(self, testloader):
        (preds, labels) = self._btfl_predict(testloader)
        return self._detail_from_preds(preds, labels)

    def test_metrics(self, model=None):
        testloader = self.load_test_data()
        (preds, labels) = self._btfl_predict(testloader)
        test_acc = (preds == labels).sum().item()
        test_num = len(labels)
        return (test_acc, test_num, 0.0)
