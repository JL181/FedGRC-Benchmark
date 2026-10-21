import torch
from torch.utils.data import DataLoader
from flcore.clients.clientavg import clientAVG
from utils.data_utils import read_client_data

class clientFedCOF(clientAVG):

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
            raise AttributeError('clientFedCOF needs a BaseHeadSplit model (model.base + model.head); register FedCOF with split_backbone_head(args).')
        in_features = getattr(head, 'in_features', None)
        if in_features is None:
            in_features = int(head.weight.shape[1])
        return int(in_features)

    @torch.no_grad()
    def fedcof_statistics(self):
        base = getattr(self.model, 'base', None)
        if base is None:
            raise AttributeError('clientFedCOF needs a BaseHeadSplit model (model.base + model.head); register FedCOF with split_backbone_head(args).')
        self.model.eval()
        class_sums = {}
        class_sample_counts = {}
        n_seen = 0
        for (x, y) in self.load_full_train_data():
            (x, y) = self._prepare_batch(x, y)
            features = base(x)
            feats = features.detach().cpu().double()
            labels = y.detach().cpu()
            for class_label in torch.unique(labels):
                class_id = int(class_label.item())
                class_data = feats[labels == class_label]
                if class_id not in class_sums:
                    class_sums[class_id] = torch.zeros(feats.shape[1], dtype=torch.float64)
                    class_sample_counts[class_id] = 0
                class_sums[class_id] += class_data.sum(dim=0)
                class_sample_counts[class_id] += int(class_data.size(0))
            n_seen += int(labels.shape[0])
        class_means = {class_id: class_sums[class_id] / float(class_sample_counts[class_id]) for class_id in sorted(class_sums)}
        sample_per_mean = {class_id: class_sample_counts[class_id] for class_id in sorted(class_sums)}
        statistics = {'class_means': class_means, 'sample_per_mean': sample_per_mean}
        return (statistics, n_seen)
