from functools import lru_cache
from pathlib import Path
import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

FEDRS_LABEL_MAPPING = {'Agriculture': 0, 'Bareland': 1, 'Forest': 2, 'Residential': 3, 'River': 4, 'Airport': 5, 'Beach': 6, 'Highway': 7, 'Industrial': 8, 'Port': 9, 'Overpass': 10, 'Parkinglot': 11, 'Bridge': 12, 'Mountain': 13, 'Meadow': 14}

N_CLASSES = 15

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

def fedrs_transform(size=64):
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def list_samples(root):
    samples = []
    for cls in sorted(os.listdir(root)):
        d = os.path.join(root, cls)
        if not os.path.isdir(d) or cls not in FEDRS_LABEL_MAPPING:
            continue
        lab = FEDRS_LABEL_MAPPING[cls]
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(IMG_EXT):
                samples.append((os.path.join(d, fn), lab))
    return samples


@lru_cache(maxsize=1)
def _views(root):
    rng = np.random.RandomState(0)
    result = []
    for cid in range(135):
        rows = list_samples(Path(root) / 'NIID-1' / f'client_{cid}')
        if not rows:
            raise ValueError(f'Missing FedRS NIID-1 client {cid} in {root}')
        labels = np.asarray([label for _, label in rows], dtype=np.int64)
        train, test = [], []
        for label in np.unique(labels):
            indices = np.where(labels == label)[0]
            rng.shuffle(indices)
            n = len(indices)
            count = 0 if n == 1 else max(1, min(int(round(0.25 * n)), n - 1))
            test.extend(indices[:count].tolist())
            train.extend(indices[count:].tolist())
        result.append(([rows[i] for i in sorted(train)], [rows[i] for i in sorted(test)]))
    return result


def _read(rows):
    transform = fedrs_transform(64)
    samples = []
    for path, label in rows:
        with Image.open(path) as image:
            samples.append((transform(image.convert('RGB')), torch.tensor(label, dtype=torch.int64)))
    return samples


def read_fedrs(root, client_id=None, is_train=True):
    if client_id is None:
        rows = list_samples(Path(root) / 'test_balanced')
        if not rows:
            raise ValueError('FedRS requires the official test_balanced global view')
        return _read(rows)
    if not 0 <= client_id < 135:
        raise ValueError('FedRS client ID must be in [0, 134]')
    return _read(_views(str(root))[client_id][0 if is_train else 1])
