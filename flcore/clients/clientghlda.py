import torch
from torch.utils.data import DataLoader
from flcore.clients.clientavg import clientAVG
from utils.data_utils import read_client_data

class clientGHLDA(clientAVG):

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def load_full_train_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=False, shuffle=False)

    def feature_dim(self):
        head = getattr(self.model, 'head', None)
        if head is None:
            raise AttributeError('clientGHLDA needs a BaseHeadSplit model (model.base + model.head); register GHLDA with split_backbone_head(args).')
        in_features = getattr(head, 'in_features', None)
        if in_features is None:
            in_features = int(head.weight.shape[1])
        return int(in_features)

    @torch.no_grad()
    def ghlda_statistics(self, num_classes=None):
        num_classes = self.num_classes if num_classes is None else int(num_classes)
        base = getattr(self.model, 'base', None)
        if base is None:
            raise AttributeError('clientGHLDA needs a BaseHeadSplit model (model.base + model.head); register GHLDA with split_backbone_head(args).')
        self.model.eval()
        C = num_classes
        d = self.feature_dim()
        A = torch.zeros(C, d, dtype=torch.float64)
        N = torch.zeros(C, dtype=torch.long)
        B = torch.zeros(d, d, dtype=torch.float64)
        seen = 0
        for (x, y) in self.load_full_train_data():
            (x, y) = self._prepare_batch(x, y)
            x32 = base(x).float()
            x64 = x32.detach().cpu().double()
            y_cpu = y.detach().cpu().long()
            B += x64.t().mm(x64)
            for cls in y_cpu.unique():
                ci = int(cls.item())
                mask = y_cpu == cls
                if mask.any():
                    xc = x64[mask]
                    A[ci] += xc.sum(dim=0)
                    N[ci] += int(mask.sum().item())
            seen += int(y_cpu.shape[0])
        return (N, A, B, seen)
