import time
import torch
import torch.nn.functional as F
from flcore.clients.clientfedcof import clientFedCOF
from flcore.servers.serverclosedform import ClosedFormHeadServer, resolve_head_impl

def shrink_cov(cov, gamma=1):
    iden = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    cov_ = cov + gamma * iden
    return cov_

def get_classifier_weights(ridge_solution):
    return F.normalize(ridge_solution)

def fedcof_aggregate(client_data, n_classes, feature_space_dim, gamma):
    dtype = torch.float64
    covs = {c: [] for c in range(n_classes)}
    class_mean_clients = {c: [] for c in range(n_classes)}
    class_mean_sums = {c: torch.zeros(size=(feature_space_dim,), dtype=dtype) for c in range(n_classes)}
    class_counts = {c: [] for c in range(n_classes)}
    class_sample_counts = {c: 0 for c in range(n_classes)}
    num_x = 0
    for (user_id, user_data) in client_data.items():
        class_means = user_data['class_means']
        sample_counts = user_data['sample_per_mean']
        for (class_id, class_mean) in class_means.items():
            class_mean_sums[class_id] += class_mean * float(sample_counts[class_id])
            class_sample_counts[class_id] += sample_counts[class_id]
            class_mean_clients[class_id].append(class_mean)
            class_counts[class_id].append(float(sample_counts[class_id]))
    class_means = {class_id: class_mean_sums[class_id] / class_sample_counts[class_id] for class_id in class_mean_sums}
    sorted_class_means = dict(sorted(class_means.items(), key=lambda item: item[0]))
    sorted_class_means_list = list(sorted_class_means.values())
    class_means_stacked = torch.stack(sorted_class_means_list)
    all_count = 0
    g_means = 0
    for (class_id, class_mean) in class_means.items():
        if class_sample_counts[class_id] > 0:
            class_mean_clients[class_id] = torch.stack(class_mean_clients[class_id])
            class_counts[class_id] = torch.tensor(class_counts[class_id], dtype=dtype)
            all_count += class_sample_counts[class_id]
            g_means += class_means[class_id] * class_sample_counts[class_id]
            mu_x = class_mean_clients[class_id] - class_means[class_id]
            mu_x *= torch.sqrt(class_counts[class_id])[:, None]
            inside = torch.mm(mu_x.T, mu_x)
            if len(class_mean_clients[class_id]) > 1:
                delta = inside / (len(class_mean_clients[class_id]) - 1)
            else:
                delta = inside / len(class_mean_clients[class_id])
            covs[class_id] = shrink_cov(delta, gamma=gamma)
            num_x += len(class_mean_clients[class_id])
    num_total = num_x
    g_means = g_means / all_count
    t1 = all_count * torch.mm(g_means.unsqueeze(1), g_means.unsqueeze(0))
    est_gram = torch.zeros(feature_space_dim, feature_space_dim, dtype=dtype)
    for label_idx in range(n_classes):
        if class_sample_counts[label_idx] > 0:
            est_gram += (class_sample_counts[label_idx] - 1) * covs[label_idx]
    est_gram = est_gram + t1
    inv_G = torch.linalg.pinv(est_gram)
    ridge_solution = []
    for label_idx in range(n_classes):
        ridge_solution.append(torch.matmul(inv_G, class_mean_sums[label_idx].unsqueeze(0).T))
    ridge_solution = torch.stack(ridge_solution).squeeze(2)
    return (ridge_solution, class_means_stacked, num_total)

def resolve_head(model):
    return resolve_head_impl(model, 'FedCOF needs a model with a classifier head (BaseHeadSplit.head / .fc / .classifier); register FedCOF with split_backbone_head(args).')

class FedCOF(ClosedFormHeadServer):
    display_name = 'FedCOF'

    def __init__(self, args, times):
        super().__init__(args, times, clientFedCOF)
        self.fcof_gamma = float(getattr(args, 'fcof_gamma', 1.0))

    @torch.no_grad()
    def fit_closed_form_head(self, model=None, clients=None):
        src = self.global_model if model is None else model
        fit_clients = list(self.clients if clients is None else clients)
        if len(fit_clients) == 0:
            raise RuntimeError('FedCOF closed-form fit needs at least one client.')
        src.eval()
        dim = self.feature_dim(src)
        num_classes = int(self.num_classes)
        start = time.time()
        client_data = {}
        total_samples = 0
        for (idx, client) in enumerate(fit_clients, start=1):
            client.set_parameters(src)
            self.add_comm_bytes(self.payload_num_bytes(src))
            (statistics, n_seen) = client.fedcof_statistics()
            client_data[client.id] = statistics
            total_samples += n_seen
            self.add_comm_bytes(self.payload_num_bytes(statistics))
            if idx % 20 == 0 or idx == len(fit_clients):
                print(f'[FedCOF] statistics {idx}/{len(fit_clients)} clients ({total_samples} samples) {time.time() - start:.1f}s')
        (ridge_solution, _class_means, num_total) = fedcof_aggregate(client_data, num_classes, dim, self.fcof_gamma)
        W = get_classifier_weights(ridge_solution)
        bias = torch.zeros(num_classes, dtype=torch.float64)
        self.install_closed_form_head(src, W, bias)
        self.closed_form_fitted = True
        print(f'[FedCOF] closed-form head fitted: d={dim} C={num_classes} gamma={self.fcof_gamma} samples={total_samples} means_shared={num_total} ({time.time() - start:.1f}s)')
        return (W, bias)

    @torch.no_grad()
    def install_closed_form_head(self, model, W, bias):
        head = resolve_head(model)
        weight = W.contiguous().to(device=self.device, dtype=torch.float32)
        bias_t = bias.contiguous().to(device=self.device, dtype=torch.float32)
        if hasattr(head, 'install'):
            head.install(weight, bias_t)
            return
        if tuple(head.weight.shape) != tuple(weight.shape):
            raise RuntimeError(f'FedCOF head shape mismatch: head {tuple(head.weight.shape)} vs closed-form {tuple(weight.shape)}.')
        head.weight.data.copy_(weight)
        if getattr(head, 'bias', None) is not None:
            head.bias.data.copy_(bias_t)
