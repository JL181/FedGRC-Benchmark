import torch
from flcore.clients.clientccvr import clientCCVR
from flcore.servers.serverclosedform import ClosedFormHeadServer
from flcore.heads.statistical_heads import head_ccvr

class CCVR(ClosedFormHeadServer):
    display_name = 'CCVR'

    def __init__(self, args, times):
        super().__init__(args, times, clientCCVR)
        self.ccvr_per = int(getattr(args, 'ccvr_per', 200))
        self.head_seed = int(getattr(args, 'head_seed', 1))

    @torch.no_grad()
    def fit_closed_form_head(self, model=None, clients=None):
        src = self.global_model if model is None else model
        fit_clients = list(self.clients if clients is None else clients)
        if len(fit_clients) == 0:
            raise RuntimeError('CCVR closed-form fit needs at least one client.')
        src.eval()
        S = self.collect_train_feature_bundle(src, fit_clients)
        (W, b) = head_ccvr(S, m_per=self.ccvr_per, head_seed=self.head_seed)[0]
        self.install_affine_head(src, W, b)
        self.closed_form_fitted = True
        print(f'[CCVR] closed-form head fitted: d={S.DIM} C={S.C} clients={S.K} virtual/class={self.ccvr_per} head_seed={self.head_seed}')
        return (W, b)
