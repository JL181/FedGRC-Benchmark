"""Single-run entry point for the 22 methods in FedGRC-Benchmark."""
import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
from flcore.trainmodel.models import BaseHeadSplit
from utils.output_utils import CELL_BY_DATASET, run_directory
from utils.checkpoint_utils import read_backbone_config

ROOT = Path(__file__).resolve().parent
REGISTRY = {'FedGRC': ('flcore.servers.servergrc', 'FedGRC'), 'FedAvg': ('flcore.servers.serveravg', 'FedAvg'), 'FedRep': ('flcore.servers.serverrep', 'FedRep'), 'FedROD': ('flcore.servers.serverrod', 'FedROD'), 'FedLC': ('flcore.servers.serverlc', 'FedLC'), 'FedTHE': ('flcore.servers.serverthe', 'FedTHE'), 'FedETF': ('flcore.servers.serveretf', 'FedETF'), 'FedFA': ('flcore.servers.serverfa', 'FedFA'), 'pFedFDA': ('flcore.servers.serverfda', 'pFedFDA'), 'FedVLS': ('flcore.servers.servervls', 'FedVLS'), 'FedDrPlus': ('flcore.servers.serverdrplus', 'FedDrPlus'), 'BTFL': ('flcore.servers.serverbtfl', 'BTFL'), 'HPFL': ('flcore.servers.serverhpfl', 'HPFL'), 'FedYoYo': ('flcore.servers.serveryoyo', 'FedYoYo'), 'FedLog': ('flcore.servers.serverfedlog', 'FedLog'), 'pFedGM': ('flcore.servers.serverpfedgm', 'pFedGM'), 'CCVR': ('flcore.servers.serverccvr', 'CCVR'), 'FedNCM': ('flcore.servers.serverncm', 'FedNCM'), 'Fed3R': ('flcore.servers.serverfed3r', 'Fed3R'), 'FedCGS': ('flcore.servers.servercgs', 'FedCGS'), 'FedCOF': ('flcore.servers.serverfedcof', 'FedCOF'), 'GHLDA': ('flcore.servers.serverghlda', 'GHLDA')}
SUPPORTED_ALGORITHMS = set(REGISTRY)
SUPPORTED_MODELS = {"ResNet18"}


def boolean(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in ('true', '1'):
        return True
    if str(value).lower() in ('false', '0'):
        return False
    raise argparse.ArgumentTypeError('Expected true/false or 1/0')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fedfa_mu', type=float, default=0.1)
    parser.add_argument('--fedfa_anchor_momentum', type=float, default=0.5)
    parser.add_argument('--fedfa_calibration_lr', type=float, default=0.001)
    parser.add_argument('-go', '--goal', type=str, default='261015')
    parser.add_argument('-dev', '--device', type=str, default='cuda', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('-did', '--device_id', type=str, default='0')
    parser.add_argument('-data', '--dataset', type=str, default='Cifar10/dir_a03_20c_gtest10')
    parser.add_argument('-ncl', '--num_classes', type=int, default=None)
    parser.add_argument('-m', '--model', type=str, default='ResNet18', choices=sorted(SUPPORTED_MODELS))
    parser.add_argument('-algo', '--algorithm', type=str, default='FedAvg', choices=sorted(SUPPORTED_ALGORITHMS))
    parser.add_argument('-lbs', '--batch_size', type=int, default=10)
    parser.add_argument('-lr', '--local_learning_rate', type=float, default=0.005)
    parser.add_argument('-ld', '--learning_rate_decay', type=boolean, default=False)
    parser.add_argument('-ldg', '--learning_rate_decay_gamma', type=float, default=0.99)
    parser.add_argument('-mom', '--momentum', type=float, default=None)
    parser.add_argument('-wd', '--weight_decay', type=float, default=None)
    parser.add_argument('-imgsz', '--img_size', type=int, default=None)
    parser.add_argument('-gr', '--global_rounds', type=int, default=None)
    parser.add_argument('-ls', '--local_epochs', type=int, default=1)
    parser.add_argument('-jr', '--join_ratio', type=float, default=1.0)
    parser.add_argument('-nc', '--num_clients', type=int, default=None)
    parser.add_argument('-eg', '--eval_gap', type=int, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--ckpt_every', type=int, default=0)
    parser.add_argument('--backbone_ckpt', type=str, default=None)
    parser.add_argument('--train_backbone', action='store_true')
    parser.add_argument('--head_seed', type=int, default=1)
    parser.add_argument('-lam', '--lamda', type=float, default=1.0)
    parser.add_argument('-tau', '--tau', type=float, default=1.0)
    parser.add_argument('-ple', '--plocal_epochs', type=int, default=1)
    parser.add_argument('--etf_personalization_rounds', type=int, default=20)
    parser.add_argument('--etf_personalization_inner_steps', type=int, default=3)
    parser.add_argument('--etf_proto_lr', type=float, default=0.1)
    parser.add_argument('--etf_proj_lr', type=float, default=0.05)
    parser.add_argument('--single_beta', type=int, default=1, choices=[0, 1])
    parser.add_argument('--local_beta', type=boolean, default=False)
    parser.add_argument('--the_steps', type=int, default=20)
    parser.add_argument('--the_alpha', type=float, default=0.1)
    parser.add_argument('--the_beta', type=float, default=0.1)
    parser.add_argument('--the_personal_lr', type=float, default=0.01)
    parser.add_argument('--the_personal_epochs', type=int, default=1)
    parser.add_argument('--yoyo_warmup', type=int, default=50)
    parser.add_argument('--yoyo_gamma', type=float, default=0.1)
    parser.add_argument('--yoyo_tau', type=float, default=1.5)
    parser.add_argument('--yoyo_kd_weight', type=float, default=4.0)
    parser.add_argument('--yoyo_kd_temp', type=float, default=1.5)
    parser.add_argument('--asd_aug', type=str, default='trivial', choices=['trivial', 'minimal'])
    parser.add_argument('--pfedgm_lambda', type=float, default=1.0)
    parser.add_argument('--pfedgm_cov_mode', type=str, default='grad', choices=['grad', 'identity'])
    parser.add_argument('--pfedgm_absent_proto', type=str, default='drop', choices=['drop', 'global', 'exclude'])
    parser.add_argument('--pfedgm_proto_ema', type=float, default=0.9)
    parser.add_argument('--pfedgm_aug', type=int, default=0, choices=[0, 1])
    parser.add_argument('--pfedgm_local_epochs', type=int, default=5)
    parser.add_argument('--pfedgm_p_lr', type=float, default=0.05)
    parser.add_argument('--pfedgm_lbfgs_iter', type=int, default=10)
    parser.add_argument('--pfedgm_lbfgs_rounds', type=int, default=5)
    parser.add_argument('--pfedgm_lbfgs_lr', type=float, default=0.05)
    parser.add_argument('--lc_missing_cap', type=float, default=10000.0)
    parser.add_argument('--drp_beta', type=float, default=0.9)
    parser.add_argument('--btfl_q_level', type=int, default=2)
    parser.add_argument('--btfl_kb', type=float, default=1.0)
    parser.add_argument('--btfl_sigma', type=float, default=0.01)
    parser.add_argument('--btfl_update_condition', type=float, default=5.0)
    parser.add_argument('--btfl_personal_lr', type=float, default=0.01)
    parser.add_argument('--btfl_personal_epochs', type=int, default=1)
    parser.add_argument('--btfl_ent_norm', type=float, default=4.605170185988092)
    parser.add_argument('--flog_lr', type=float, default=0.0005)
    parser.add_argument('--flog_map_iters', type=int, default=10000)
    parser.add_argument('--flog_map_lr', type=float, default=0.01)
    parser.add_argument('--flog_map_tol', type=float, default=0.001)
    parser.add_argument('--flog_stat_batch_size', type=int, default=100)
    parser.add_argument('--f3r_lambda', type=float, default=0.01)
    parser.add_argument('--fcof_gamma', type=float, default=1.0)
    parser.add_argument('--ghlda_lda_shrink', '--ghlda_shrinkage', dest='ghlda_lda_shrink', type=float, default=0.05)
    parser.add_argument('--ccvr_per', type=int, default=200)
    parser.add_argument('--cgs_variant', type=str, default='exactnorm', choices=['exactnorm'])
    parser.add_argument('--rare_class_k', type=int, default=2)
    parser.add_argument('--cgrl_lambda_dir', type=float, default=1.0)
    parser.add_argument('--cgrl_lambda_rad', type=float, default=1.0)
    parser.add_argument('--cgrl_beta', type=float, default=1.0)
    parser.add_argument('--cgrl_temperature', type=float, default=0.1)
    parser.add_argument('--cgrl_direction_eta', type=float, default=0.5)
    parser.add_argument('--cgrl_radial_eta', type=float, default=0.5)
    parser.add_argument('--cgrl_warmup', type=int, default=50)
    parser.add_argument('--cgrl_eps_r', type=float, default=1e-06)
    parser.add_argument('--cgrl_rmin', type=float, default=0.001)
    parser.add_argument('--mms_ridge_factor', type=float, default=0.1)
    parser.add_argument('--sasp_m0', type=float, default=64.0)
    parser.add_argument('--grc_feature_batch_size', type=int, default=256)
    parser.add_argument('--hpfl_backbone_source', choices=['FedAvg', 'FedYoYo'])
    parser.add_argument('--hpfl_head_epochs', type=int, default=10)
    parser.add_argument('--hpfl_head_lr', type=float, default=0.1)
    parser.add_argument('--hpfl_head_batch_size', type=int, default=128)
    parser.add_argument('--hpfl_head_momentum', type=float, default=0.0)
    parser.add_argument('--hpfl_head_weight_decay', type=float, default=0.0001)
    parser.add_argument('--hpfl_feature_batch_size', type=int, default=10)
    parser.add_argument('--hpfl_query_batch_size', type=int, default=128)
    parser.add_argument('--hpfl_query_seed', type=int, default=1)
    parser.add_argument('--hpfl_marker_limit', type=int, default=0)
    parser.add_argument('--hpfl_noise_coef', type=float, default=0.0)
    parser.add_argument('--hpfl_mmd_block_size', type=int, default=512)
    parser.add_argument('--data_root', type=Path, default=ROOT / 'dataset')
    parser.add_argument('--output_root', type=Path, default=ROOT / 'results')
    return parser

def split_backbone_head(args):
    if not hasattr(args.model, 'fc'):
        raise NotImplementedError(f'{args.algorithm} requires a model with an fc classifier to split.')
    args.head = copy.deepcopy(args.model.fc)
    args.model.fc = nn.Identity()
    args.model = BaseHeadSplit(args.model, args.head)

def split_backbone_head_etf(args):
    if not hasattr(args.model, 'fc'):
        raise NotImplementedError(f'FedETF requires a model with an fc classifier to split.')
    from flcore.trainmodel.etf_head import ETFHead
    backbone_out_dim = args.model.fc.in_features
    args.model.fc = nn.Identity()
    args.head = ETFHead(backbone_out_dim, args.num_classes).to(args.device)
    args.model = BaseHeadSplit(args.model, args.head)

def split_backbone_head_pfedgm(args):
    if not hasattr(args.model, 'fc'):
        raise NotImplementedError('pFedGM requires a model with an fc classifier to split.')
    from flcore.trainmodel.pfedgm_head import NavigatorHead
    backbone_out_dim = args.model.fc.in_features
    args.model.fc = nn.Identity()
    args.head = NavigatorHead(backbone_out_dim, args.num_classes).to(args.device)
    args.model = BaseHeadSplit(args.model, args.head)

INTERNAL_DEFAULTS = {'client_drop_rate': 0.0, 'train_slow_rate': 0.0, 'send_slow_rate': 0.0, 'time_threthold': 10000, 'random_join_ratio': False, 'save_eval_detail': True}


def resolve_args(args):
    for key, value in INTERNAL_DEFAULTS.items():
        setattr(args, key, value)
    if args.dataset not in CELL_BY_DATASET:
        raise ValueError('Choose one of the four bundled CIFAR splits or FedRS')
    is_fedrs = args.dataset == 'FedRS'
    classes = 15 if is_fedrs else (100 if args.dataset.startswith('Cifar100/') else 10)
    expected = dict(num_classes=classes, num_clients=135 if is_fedrs else 20, img_size=64 if is_fedrs else 32)
    for key, value in expected.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
        elif getattr(args, key) != value:
            raise ValueError(f'{args.dataset} requires {key}={value}')
    if args.global_rounds is None:
        args.global_rounds = 500 if is_fedrs else (300 if classes == 100 else 200)
    if args.eval_gap is None:
        args.eval_gap = 20 if is_fedrs else (6 if classes == 100 else 5)
    if args.backbone_ckpt:
        args.backbone_ckpt = str(Path(args.backbone_ckpt).resolve())
    if args.backbone_ckpt and args.train_backbone:
        raise ValueError('Choose exactly one of --backbone_ckpt and --train_backbone')
    head_methods = {'CCVR', 'FedNCM', 'Fed3R', 'FedCGS', 'FedCOF', 'GHLDA', 'HPFL'}
    if (args.backbone_ckpt or args.train_backbone) and args.algorithm not in head_methods:
        raise ValueError('This method trains its own representation')
    if args.algorithm in head_methods and not (args.backbone_ckpt or args.train_backbone):
        raise ValueError('Classifier methods require --backbone_ckpt or --train_backbone')
    if args.algorithm == 'HPFL' and not args.backbone_ckpt:
        raise ValueError('HPFL requires --backbone_ckpt')
    if args.backbone_ckpt:
        args.global_rounds = 0
    if args.grc_feature_batch_size < 1 or args.cgrl_warmup < 1 or args.batch_size < 1 or args.local_epochs < 1 or args.global_rounds < 0 or args.eval_gap < 1:
        raise ValueError('Invalid training budget')
    if not 0 < args.join_ratio <= 1 or int(args.num_clients * args.join_ratio) < 1:
        raise ValueError('The participation ratio must select at least one client')
    if not args.goal or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in args.goal):
        raise ValueError('--goal must contain only letters, digits, dots, hyphens or underscores')
    for key in ('cgrl_lambda_dir', 'cgrl_lambda_rad', 'cgrl_beta', 'cgrl_temperature', 'cgrl_eps_r', 'cgrl_rmin', 'mms_ridge_factor', 'sasp_m0'):
        value = getattr(args, key)
        if not np.isfinite(value) or value < 0 or (key in ('cgrl_temperature', 'cgrl_eps_r', 'mms_ridge_factor', 'sasp_m0') and value == 0):
            raise ValueError(f'Invalid {key}')
    for key in ('cgrl_direction_eta', 'cgrl_radial_eta'):
        if not 0 <= getattr(args, key) <= 1:
            raise ValueError(f'{key} must be in [0, 1]')
    return args


def run(args):
    args = resolve_args(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; choose --device cpu explicitly')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ['FEDGRC_DATA_ROOT'] = str(args.data_root.resolve())
    args.backbone_source = None
    if args.backbone_ckpt:
        source_config = read_backbone_config(args.backbone_ckpt, args)
        args.backbone_source = source_config['algorithm']
        args.backbone_goal = source_config.get('goal')
        from hashlib import sha256
        digest = sha256()
        with open(args.backbone_ckpt, 'rb') as checkpoint:
            for chunk in iter(lambda: checkpoint.read(1 << 20), b''):
                digest.update(chunk)
        args.backbone_sha256 = digest.hexdigest()
    folder = run_directory(args)
    folder.mkdir(parents=True, exist_ok=False)
    args.run_dir = folder
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (folder / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    args.model = resnet18(weights=None, num_classes=args.num_classes).to(args.device)
    if args.algorithm == 'FedETF':
        split_backbone_head_etf(args)
    elif args.algorithm == 'pFedGM':
        split_backbone_head_pfedgm(args)
    else:
        split_backbone_head(args)
    module, name = REGISTRY[args.algorithm]
    server = getattr(importlib.import_module(module), name)(args, 0)
    server.train()
    metrics = {key: float(value) if np.isfinite(value) else None for key, value in server.last_metrics.items()}
    (folder / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
    print(f'Completed {args.algorithm}; output: {folder}')


if __name__ == '__main__':
    run(build_parser().parse_args())
