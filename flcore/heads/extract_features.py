import numpy as np
import torch
from utils.data_utils import read_client_data

@torch.no_grad()
def feats_l4(model, data, dev, bs=256):
    (F, Y) = ([], [])
    for i in range(0, len(data), bs):
        xb = torch.stack([data[j][0] for j in range(i, min(i + bs, len(data)))]).to(dev)
        rep = model.base(xb)
        F.append(rep.cpu().numpy())
        Y.append(np.array([int(data[j][1]) for j in range(i, min(i + bs, len(data)))]))
    return (np.concatenate(F).astype(np.float32), np.concatenate(Y).astype(np.int64))

@torch.no_grad()
def extract_client_train_features(model, dataset, client_id, dev, bs=256):
    data = read_client_data(dataset, client_id, is_train=True)
    return feats_l4(model, data, dev, bs=bs)

from contextlib import contextmanager
import random
import hashlib
import json
import torch.nn.functional as F

def model_hash(model):
    digest = hashlib.sha256()
    for (name, value) in sorted(model.state_dict().items()):
        t = value.detach().cpu().contiguous()
        meta = json.dumps([name, str(t.dtype), list(t.shape)], separators=(',', ':')).encode()
        digest.update(len(meta).to_bytes(8, 'big'))
        digest.update(meta)
        digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

@contextmanager
def isolated_eval(model):
    modes = [(module, module.training) for module in model.modules()]
    (py_rng, np_rng) = (random.getstate(), np.random.get_state())
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    mps_rng = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            model.eval()
            with torch.no_grad():
                yield
    finally:
        for (module, training) in modes:
            module.training = training
        random.setstate(py_rng)
        np.random.set_state(np_rng)
        if mps_rng is not None:
            torch.mps.set_rng_state(mps_rng)

@contextmanager
def pooled_features(model):
    (acts, handles) = ({}, [])

    def capture(name):

        def hook(_module, _inputs, output):
            acts[name] = F.adaptive_avg_pool2d(output, 1).flatten(1)
        return hook
    try:
        for name in ('layer2', 'layer3'):
            handles.append(getattr(model.base, name).register_forward_hook(capture(name)))

        def forward(x):
            acts.clear()
            h4 = model.base(x)
            blocks = (acts.pop('layer2'), acts.pop('layer3'), h4)
            return (torch.cat(blocks, dim=1), h4)
        yield forward
    finally:
        for handle in handles:
            handle.remove()
        acts.clear()
