import torch
from flcore.clients.clientcgs import clientCGS
from flcore.servers.serverclosedform import ClosedFormHeadServer
from flcore.heads.statistical_heads import head_fedcgs_exactnorm

class FedCGS(ClosedFormHeadServer):
    display_name = 'FedCGS'

    def __init__(self, args, times):
        super().__init__(args, times, clientCGS)
        self.cgs_variant = str(getattr(args, 'cgs_variant', 'exactnorm'))

    @torch.no_grad()
    def fit_closed_form_head(self, model=None, clients=None):
        src = self.global_model if model is None else model
        fit_clients = list(self.clients if clients is None else clients)
        if len(fit_clients) == 0:
            raise RuntimeError('FedCGS closed-form fit needs at least one client.')
        src.eval()
        S = self.collect_train_feature_bundle(src, fit_clients)
        (W, b) = head_fedcgs_exactnorm(S)[0]
        self.install_affine_head(src, W, b)
        self.closed_form_fitted = True
        print(f'[FedCGS] closed-form head fitted: d={S.DIM} C={S.C} clients={S.K} variant={self.cgs_variant} (total-cov + log-prior)')
        return (W, b)
