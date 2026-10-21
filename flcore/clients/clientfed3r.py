import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from flcore.clients.clientavg import clientAVG
from utils.data_utils import read_client_data

class clientFed3R(clientAVG):

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
            raise AttributeError('clientFed3R needs a BaseHeadSplit model (model.base + model.head); register Fed3R with split_backbone_head(args).')
        in_features = getattr(head, 'in_features', None)
        if in_features is None:
            in_features = int(head.weight.shape[1])
        return int(in_features)

    @torch.no_grad()
    def fed3r_statistics(self, num_classes=None):
        num_classes = self.num_classes if num_classes is None else int(num_classes)
        base = getattr(self.model, 'base', None)
        if base is None:
            raise AttributeError('clientFed3R needs a BaseHeadSplit model (model.base + model.head); register Fed3R with split_backbone_head(args).')
        self.model.eval()
        dim = self.feature_dim()
        A = torch.zeros((dim + 1, dim + 1), dtype=torch.float32, device=self.device)
        b = torch.zeros((dim + 1, num_classes), dtype=torch.float32, device=self.device)
        seen = 0
        for (x, y) in self.load_full_train_data():
            (x, y) = self._prepare_batch(x, y)
            features = base(x)
            feats_with_bias = torch.cat((features, torch.ones((features.shape[0], 1), dtype=torch.float32, device=features.device)), dim=1)
            feats_with_bias = feats_with_bias.detach()
            one_hot_encoded = F.one_hot(y, num_classes=num_classes)
            one_hot_encoded = one_hot_encoded.float()
            Ak = feats_with_bias.T @ feats_with_bias
            bk = feats_with_bias.T @ one_hot_encoded
            A += Ak
            b += bk
            seen += int(y.shape[0])
        return (A.cpu().double(), b.cpu().double(), seen)
