import time
import torch
from flcore.clients.clientfed3r import clientFed3R
from flcore.servers.serverclosedform import ClosedFormHeadServer, resolve_head_impl

def calc_ridge_params(A, b):
    W = torch.linalg.solve(A, b)
    bias = W[-1, :]
    W = W[:-1, :]
    return (W, bias)

def normalize_W(W, bias, warnings=True):
    norm = torch.norm(W, dim=0, keepdim=True)
    if torch.any(norm == 0.0):
        if warnings:
            print('WARNING: 0 encountered in norm, substituting with 1e-6')
        norm[norm == 0.0] = 1e-06
    W = W / norm
    bias = bias / norm[0, :]
    return (W, bias)

def resolve_head(model):
    return resolve_head_impl(model, 'Fed3R needs a model with a classifier head (BaseHeadSplit.head / .fc / .classifier); register Fed3R with split_backbone_head(args).')

class Fed3R(ClosedFormHeadServer):
    display_name = 'Fed3R'

    def __init__(self, args, times):
        super().__init__(args, times, clientFed3R)
        self.f3r_lambda = float(getattr(args, 'f3r_lambda', 0.01))

    @torch.no_grad()
    def fit_closed_form_head(self, model=None, clients=None):
        src = self.global_model if model is None else model
        fit_clients = list(self.clients if clients is None else clients)
        if len(fit_clients) == 0:
            raise RuntimeError('Fed3R closed-form fit needs at least one client.')
        src.eval()
        dim = self.feature_dim(src)
        num_classes = int(self.num_classes)
        A = self.f3r_lambda * torch.eye(dim + 1, dtype=torch.float64)
        b = torch.zeros((dim + 1, num_classes), dtype=torch.float64)
        start = time.time()
        total_samples = 0
        for (idx, client) in enumerate(fit_clients, start=1):
            client.set_parameters(src)
            self.add_comm_bytes(self.payload_num_bytes(src))
            (Ak, bk, nk) = client.fed3r_statistics(num_classes=num_classes)
            A += Ak
            b += bk
            total_samples += nk
            self.add_comm_bytes(self.payload_num_bytes(Ak) + self.payload_num_bytes(bk))
            if idx % 20 == 0 or idx == len(fit_clients):
                print(f'[Fed3R] statistics {idx}/{len(fit_clients)} clients ({total_samples} samples) {time.time() - start:.1f}s')
        (W, bias) = normalize_W(*calc_ridge_params(A, b))
        self.install_closed_form_head(src, W, bias)
        self.closed_form_fitted = True
        print(f'[Fed3R] closed-form head fitted: d={dim} C={num_classes} lambda={self.f3r_lambda} samples={total_samples} ({time.time() - start:.1f}s)')
        return (W, bias)

    @torch.no_grad()
    def install_closed_form_head(self, model, W, bias):
        head = resolve_head(model)
        weight = W.T.contiguous().to(device=self.device, dtype=torch.float32)
        bias_t = bias.contiguous().to(device=self.device, dtype=torch.float32)
        if hasattr(head, 'install'):
            head.install(weight, bias_t)
            return
        if getattr(head, 'bias', None) is None:
            raise RuntimeError('Fed3R requires a classifier head with a bias term.')
        if tuple(head.weight.shape) != tuple(weight.shape):
            raise RuntimeError(f'Fed3R head shape mismatch: head {tuple(head.weight.shape)} vs closed-form {tuple(weight.shape)}.')
        head.weight.data.copy_(weight)
        head.bias.data.copy_(bias_t)
