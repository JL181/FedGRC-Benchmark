import torch

def _distance(x, y):
    return torch.cdist(x, y, compute_mode='donot_use_mm_for_euclid_dist').square()

@torch.no_grad()
def biased_mmd(x, y, block_size=512):
    if x.ndim != 2 or y.ndim != 2 or (not len(x)) or (not len(y)) or (x.shape[1] != y.shape[1]):
        raise ValueError('MMD needs two nonempty [samples, features] arrays')
    if block_size < 1:
        raise ValueError('block_size must be positive')
    (x, y) = (x.to(dtype=torch.float64), y.to(device=x.device, dtype=torch.float64))
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('Non-finite MMD features')
    z = torch.cat((x, y))
    bandwidth = (2 * (z - z.mean(0)).square().sum() / (len(z) - 1)).clamp_min(torch.finfo(z.dtype).tiny)
    scales = bandwidth * z.new_tensor([0.25, 0.5, 1.0, 2.0, 4.0])

    def kernel_mean(a, b):
        total = z.new_zeros(())
        for i in range(0, len(a), block_size):
            for j in range(0, len(b), block_size):
                d = _distance(a[i:i + block_size], b[j:j + block_size])
                total += torch.exp(-d.unsqueeze(0) / scales[:, None, None]).sum()
        return total / (len(a) * len(b))
    return kernel_mean(x, x) - 2 * kernel_mean(x, y) + kernel_mean(y, y)
import hashlib
import numpy as np
import torch

def stream_seed(seed, domain, client_id=0):
    token = f'HPFL:{seed}:{domain}:{client_id}'.encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], 'little') % (2 ** 63 - 1)

def query_manifest(size, batch_size, seed, view):
    if batch_size < 1 or size < 0:
        raise ValueError('Invalid query size')
    rng = np.random.default_rng(stream_seed(seed, view))
    order = rng.permutation(size)
    return [order[i:i + batch_size] for i in range(0, size, batch_size)]

def build_marker(features, seed, client_id, limit=0, noise_coef=0.0):
    x = features.detach().cpu().clone()
    if not len(x) or limit < 0 or noise_coef < 0:
        raise ValueError('Invalid marker configuration')
    generator = torch.Generator().manual_seed(stream_seed(seed, 'marker', client_id))
    if noise_coef:
        noise = torch.randn(x.shape, generator=generator, dtype=x.dtype)
        noise = noise * (x.var(0, unbiased=False) + 0.2).sqrt() + x.mean(0)
        x = (x + noise_coef * noise) / (1 + noise_coef)
    if limit and len(x) > limit:
        x = x[torch.randperm(len(x), generator=generator)[:limit]]
    return x.contiguous()

class HPFLPredictor:

    def __init__(self, shared_head, heads, markers, device='cpu', block_size=512, head_batch_size=10):
        if not heads or set(heads) != set(markers):
            raise ValueError('Each candidate must have exactly one head and marker')
        self.shared_head = shared_head
        self.heads = heads
        self.markers = markers
        self.device = torch.device(device)
        self.block_size = block_size
        self.head_batch_size = head_batch_size
        if head_batch_size < 1:
            raise ValueError('head_batch_size must be positive')

    @torch.no_grad()
    def _predict_head(self, head, features):
        head = head.to(self.device).eval()
        result = []
        for i in range(0, len(features), self.head_batch_size):
            result.append(head(features[i:i + self.head_batch_size].to(self.device)).argmax(1).cpu())
        head.cpu()
        return torch.cat(result).numpy() if result else np.empty(0, dtype=np.int64)

    @torch.no_grad()
    def predict(self, features, client_id, state, batches):
        if state == 'before':
            return (self._predict_head(self.shared_head, features), [])
        if state == 'personalized':
            return (self._predict_head(self.heads[client_id], features), [])
        if state != 'selected':
            raise ValueError(f'Unknown HPFL state: {state}')
        flat = np.concatenate(batches) if batches else np.empty(0, dtype=np.int64)
        if not np.array_equal(np.sort(flat), np.arange(len(features))):
            raise ValueError('Query manifest must cover each sample exactly once')
        ids = sorted(self.heads)
        predictions = np.empty(len(features), dtype=np.int64)
        routes = []
        for indices in batches:
            query = features[torch.as_tensor(indices)].to(self.device)
            distances = [float(biased_mmd(query, self.markers[cid].to(self.device), self.block_size)) for cid in ids]
            winner = ids[int(np.argmin(distances))]
            predictions[indices] = self._predict_head(self.heads[winner], query)
            routes.append({'sample_ids': indices.tolist(), 'head_id': winner, 'candidate_ids': ids, 'mmd': distances})
        return (predictions, routes)

def score_predictions(predictions, labels, num_classes):
    (predictions, labels) = (np.asarray(predictions), np.asarray(labels))
    if predictions.shape != labels.shape:
        raise ValueError('Prediction/label length mismatch')
    cm = np.bincount(labels * num_classes + predictions, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return {'correct': int(np.trace(cm)), 'total': int(len(labels)), 'per_class_correct': np.diag(cm).copy(), 'per_class_total': cm.sum(1), 'confusion_matrix': cm}
import copy
import math
import torch
import torch.nn.functional as F

def fit_head(shared_head, features, labels, args, client_id):
    if len(features) == 0 or len(features) != len(labels) or args.hpfl_head_epochs < 1 or (args.hpfl_head_batch_size < 1):
        raise ValueError('HPFL head training requires nonempty matched features/labels and positive epochs/batch')
    head = copy.deepcopy(shared_head).to(args.device).train().requires_grad_(True)
    optimizer = torch.optim.SGD(head.parameters(), lr=args.hpfl_head_lr, momentum=args.hpfl_head_momentum, weight_decay=args.hpfl_head_weight_decay)
    generator = torch.Generator().manual_seed(stream_seed(args.seed, 'head', client_id))
    losses = []
    for epoch in range(args.hpfl_head_epochs):
        order = torch.randperm(len(features), generator=generator)
        total = 0.0
        for ids in order.split(args.hpfl_head_batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = head(features[ids].to(args.device))
            loss = F.cross_entropy(logits, labels[ids].to(args.device))
            if not torch.isfinite(loss):
                raise RuntimeError(f'HPFL non-finite head loss: client {client_id}')
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        losses.append(total / len(features))
    return (head.cpu().eval().requires_grad_(False), {'epoch_training_ce': losses, 'samples_per_epoch': len(features), 'optimizer_steps': math.ceil(len(features) / args.hpfl_head_batch_size) * args.hpfl_head_epochs, 'shuffle_seed': stream_seed(args.seed, 'head', client_id), 'drop_last': False})

class clientHPFL:

    def __init__(self, cid, train_features, train_labels, test_features, test_labels, num_classes):
        self.id = cid
        (self.train_features, self.train_labels) = (train_features, train_labels)
        (self.test_features, self.test_labels) = (test_features, test_labels)
        (self.train_samples, self.test_samples) = (len(train_labels), len(test_labels))
        self.num_classes = num_classes
        self.predictor = None
        self.state = None
        self.local_batches = None
        self.global_detail = None
        self.routes = []

    def supports_generalization_detail(self):
        return True

    @torch.no_grad()
    def train_metrics(self):
        head = self.predictor.shared_head if self.state == 'before' else self.predictor.heads[self.id]
        head = head.to(self.predictor.device).eval()
        loss = 0.0
        for ids in torch.arange(self.train_samples).split(128):
            loss += float(F.cross_entropy(head(self.train_features[ids].to(self.predictor.device)), self.train_labels[ids].to(self.predictor.device), reduction='sum'))
        head.cpu()
        return (loss, self.train_samples)

    def detailed_personalized_test_metrics(self):
        (predictions, self.routes) = self.predictor.predict(self.test_features, self.id, self.state, self.local_batches)
        return score_predictions(predictions, self.test_labels.numpy(), self.num_classes)

    def detailed_generalization_test_metrics(self, _loader):
        return self.global_detail
