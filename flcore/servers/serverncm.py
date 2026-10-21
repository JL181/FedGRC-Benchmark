import torch
from flcore.clients.clientncm import clientNCM
from flcore.servers.serverclosedform import ClosedFormHeadServer
from flcore.heads.statistical_heads import head_fedncm

class FedNCM(ClosedFormHeadServer):
    display_name = 'FedNCM'

    def __init__(self, args, times):
        super().__init__(args, times, clientNCM)

    @torch.no_grad()
    def fit_closed_form_head(self, model=None, clients=None):
        src = self.global_model if model is None else model
        fit_clients = list(self.clients if clients is None else clients)
        if len(fit_clients) == 0:
            raise RuntimeError('FedNCM closed-form fit needs at least one client.')
        src.eval()
        S = self.collect_train_feature_bundle(src, fit_clients)
        (W, b) = head_fedncm(S)[0]
        self.install_affine_head(src, W, b)
        self.closed_form_fitted = True
        print(f'[FedNCM] closed-form head fitted: d={S.DIM} C={S.C} clients={S.K} samples={len(S.pty)} (zero-bias NCM)')
        return (W, b)
