import time
import torch
from flcore.clients.clientghlda import clientGHLDA
from flcore.servers.serverclosedform import ClosedFormHeadServer, resolve_head_impl
MIN_EPS = 1e-06

def symmetrize(S):
    return 0.5 * (S + S.t())

def shrink_cov(S, alpha):
    S = symmetrize(S)
    d = S.shape[0]
    tr = torch.trace(S).clamp_min(0.0)
    return (1.0 - alpha) * S + alpha * (tr / max(1, d)) * torch.eye(d, dtype=S.dtype, device=S.device)

def aggregate_clients_lda(A_sum, N_sum, B_sum, num_classes):
    if A_sum is None:
        raise RuntimeError('No valid client stats found.')
    (C, d) = A_sum.shape
    assert C == num_classes, f'Stats num_classes mismatch: stats C={C}, config C={num_classes}'
    N_total = int(N_sum.sum().item())
    mu = A_sum / N_sum.clamp_min(1).unsqueeze(1)
    pri = (N_sum.to(torch.float64) / float(max(1, N_total))).clamp_min(1e-12)
    logp = torch.log(pri)
    SW = B_sum.clone()
    for c in range(C):
        Nc = int(N_sum[c].item())
        if Nc > 0:
            mc = mu[c].unsqueeze(1)
            SW -= float(Nc) * (mc @ mc.t())
    denom = float(max(1, N_total - C))
    Sigma_pool = symmetrize(SW / denom)
    return dict(C=C, d=d, N_total=N_total, mu=mu, pri=pri, logp=logp, Sigma_pool=Sigma_pool)

def lda_head_params(mu64, Sigma_pool64, logp64, lda_shrink):
    Sig = shrink_cov(Sigma_pool64, lda_shrink)
    Sig = symmetrize(Sig) + MIN_EPS * torch.eye(Sig.shape[0], dtype=torch.float64, device=Sig.device)
    S = shrink_cov(Sig, 0.0)
    S = symmetrize(S) + MIN_EPS * torch.eye(S.shape[0], dtype=torch.float64, device=S.device)
    W = torch.linalg.solve(S, mu64.t())
    b = -0.5 * (mu64 * W.t()).sum(dim=1) + logp64
    return (W, b, S)

def resolve_head(model):
    return resolve_head_impl(model, 'GHLDA needs a model with a classifier head (BaseHeadSplit.head / .fc / .classifier); register GHLDA with split_backbone_head(args).')

class GHLDA(ClosedFormHeadServer):
    display_name = 'GH-LDA'

    def __init__(self, args, times):
        super().__init__(args, times, clientGHLDA)
        self.ghlda_lda_shrink = float(getattr(args, 'ghlda_lda_shrink', 0.05))

    @torch.no_grad()
    def fit_closed_form_head(self, model=None, clients=None):
        src = self.global_model if model is None else model
        fit_clients = list(self.clients if clients is None else clients)
        if len(fit_clients) == 0:
            raise RuntimeError('GH-LDA closed-form fit needs at least one client.')
        src.eval()
        dim = self.feature_dim(src)
        num_classes = int(self.num_classes)
        A_sum = torch.zeros((num_classes, dim), dtype=torch.float64)
        N_sum = torch.zeros(num_classes, dtype=torch.long)
        B_sum = torch.zeros((dim, dim), dtype=torch.float64)
        start = time.time()
        total_samples = 0
        for (idx, client) in enumerate(fit_clients, start=1):
            client.set_parameters(src)
            self.add_comm_bytes(self.payload_num_bytes(src))
            (Nk, Ak, Bk, nk) = client.ghlda_statistics(num_classes=num_classes)
            A_sum += Ak
            N_sum += Nk
            B_sum += Bk
            total_samples += nk
            self.add_comm_bytes(self.payload_num_bytes(Nk) + self.payload_num_bytes(Ak) + self.payload_num_bytes(Bk))
            if idx % 20 == 0 or idx == len(fit_clients):
                print(f'[GH-LDA] statistics {idx}/{len(fit_clients)} clients ({total_samples} samples) {time.time() - start:.1f}s')
        agg = aggregate_clients_lda(A_sum, N_sum, B_sum, num_classes=num_classes)
        (W, b, _) = lda_head_params(agg['mu'], agg['Sigma_pool'], agg['logp'], self.ghlda_lda_shrink)
        self.install_closed_form_head(src, W, b)
        self.closed_form_fitted = True
        absent = int((N_sum == 0).sum().item())
        print(f"[GH-LDA] closed-form head fitted: d={dim} C={num_classes} lda_shrink={self.ghlda_lda_shrink} N_total={agg['N_total']} globally-absent classes={absent} ({time.time() - start:.1f}s)")
        return (W, b)

    @torch.no_grad()
    def install_closed_form_head(self, model, W, b):
        head = resolve_head(model)
        weight = W.t().contiguous().to(device=self.device, dtype=torch.float32)
        bias_t = b.contiguous().to(device=self.device, dtype=torch.float32)
        if hasattr(head, 'install'):
            head.install(weight, bias_t)
            return
        if getattr(head, 'bias', None) is None:
            raise RuntimeError('GH-LDA installs an intercept (official GH_OFL_server.py:406) but the head has no bias term.')
        if tuple(head.weight.shape) != tuple(weight.shape):
            raise RuntimeError(f'GH-LDA head shape mismatch: head {tuple(head.weight.shape)} vs closed-form {tuple(weight.shape)}.')
        head.weight.data.copy_(weight)
        head.bias.data.copy_(bias_t)
