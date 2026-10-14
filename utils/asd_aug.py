import torch
import torchvision.transforms as T
_STATS = {'cifar10': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 'cifar100': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 'fedrs': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))}

def _resolve_stats(dataset):
    name = dataset.lower()
    best = None
    for (key, val) in _STATS.items():
        if key in name and (best is None or len(key) > len(best[0])):
            best = (key, val)
    if best is not None:
        return best[1]
    return ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

class Cutout:

    def __init__(self, length=16):
        self.length = length

    def __call__(self, img):
        (h, w) = (img.shape[-2], img.shape[-1])
        y = int(torch.randint(0, h, (1,)).item())
        x = int(torch.randint(0, w, (1,)).item())
        y1 = max(0, y - self.length // 2)
        y2 = min(h, y + self.length // 2)
        x1 = max(0, x - self.length // 2)
        x2 = min(w, x + self.length // 2)
        img = img.clone()
        img[..., y1:y2, x1:x2] = 0
        return img

class ASDAug:

    def __init__(self, dataset, img_size=32, per_sample=False):
        (mean, std) = _resolve_stats(dataset)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
        self.per_sample = per_sample
        self._ms_cache = {}
        self.weak = T.Compose([T.RandomCrop(img_size, padding=4), T.RandomHorizontalFlip(), T.RandomRotation(15)])
        self.strong = T.Compose([T.RandomCrop(img_size, padding=4), T.RandomHorizontalFlip(), T.TrivialAugmentWide(), Cutout(16)])

    def _ms(self, device):
        key = str(device)
        if key not in self._ms_cache:
            self._ms_cache[key] = (self.mean.to(device), self.std.to(device))
        return self._ms_cache[key]

    def _apply(self, x, tfm):
        (m, s) = self._ms(x.device)
        u8 = ((x.detach() * s + m).clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        if self.per_sample:
            u_out = torch.stack([tfm(u8[i]) for i in range(u8.shape[0])], dim=0)
        else:
            u_out = tfm(u8)
        x01 = u_out.to(dtype=x.dtype) / 255.0
        return (x01 - m) / s

    def weak_view(self, x):
        return self._apply(x, self.weak)

    def strong_view(self, x):
        return self._apply(x, self.strong)
