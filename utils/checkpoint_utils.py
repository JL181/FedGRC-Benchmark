import json
from pathlib import Path
import torch


def read_backbone_config(path, args):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Missing backbone checkpoint: {path}')
    config_path = path.parent / 'config.json'
    if not config_path.is_file():
        raise ValueError(f'Missing checkpoint configuration: {config_path}. Use a backbone produced by this repository.')
    config = json.loads(config_path.read_text())
    for key in ('dataset', 'seed', 'num_classes', 'num_clients', 'img_size'):
        if config.get(key) != getattr(args, key):
            raise ValueError(f'Backbone {key}={config.get(key)!r} does not match {getattr(args, key)!r}')
    if config.get('algorithm') not in ('FedAvg', 'FedYoYo'):
        raise ValueError('Attach requires a FedAvg or FedYoYo backbone')
    requested = getattr(args, 'hpfl_backbone_source', None) if args.algorithm == 'HPFL' else None
    if requested is not None and config['algorithm'] != requested:
        raise ValueError('HPFL backbone source differs from checkpoint configuration')
    return config


def load_backbone(path, reference, args):
    config = read_backbone_config(path, args)
    source = getattr(args, 'backbone_source', None)
    if source is not None and config['algorithm'] != source:
        raise ValueError('Backbone source changed after run setup')
    loaded = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(loaded, torch.nn.Module):
        raise ValueError('Expected a terminal model checkpoint')
    expected, actual = reference.state_dict(), loaded.state_dict()
    if list(expected) != list(actual):
        raise ValueError('Backbone parameter/buffer keys differ')
    for key in expected:
        value = actual[key]
        if value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise ValueError(f'Backbone shape/dtype mismatch: {key}')
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f'Non-finite backbone state: {key}')
    return loaded
