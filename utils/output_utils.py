from datetime import datetime
from pathlib import Path
import json

CELL_BY_DATASET = {'Cifar10/dir_a03_20c_gtest10': 'c10d03', 'Cifar10/dir_a01_20c_gtest10': 'c10d01', 'Cifar100/dir_a03_20c_gtest10': 'c100d03', 'Cifar100/dir_a01_20c_gtest10': 'c100d01', 'FedRS': 'fedrs'}


def run_name(args):
    source = getattr(args, 'backbone_source', None)
    method = f'{source}_{args.algorithm}' if source else args.algorithm
    return f'{method}_{args.goal}'


def run_directory(args):
    return Path(args.output_root).resolve() / CELL_BY_DATASET[args.dataset] / f'seed{args.seed}' / run_name(args)


def data_dir(args):
    path = Path(args.run_dir) / 'data'
    path.mkdir(exist_ok=True)
    return str(path)


def round_ckpt_dir(args):
    path = Path(args.run_dir) / 'checkpoints'
    path.mkdir(exist_ok=True)
    return str(path)


def flat_model_basename(args):
    return f'{run_name(args)}_seed{args.seed}'


SUMMARY_KIND = 'summary'
DETAIL_KIND = 'detail'
RESULT_FORMAT_VERSION = '1.1'


def result_file_path(args, artifact_kind):
    if artifact_kind not in (SUMMARY_KIND, DETAIL_KIND):
        raise ValueError(f'Unknown evaluation artifact kind: {artifact_kind}')
    return str(Path(data_dir(args)) / f'{artifact_kind}.h5')


def current_timestamp():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def _sanitize_metadata_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_sanitize_metadata_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _sanitize_metadata_value(item) for key, item in value.items()}
    return repr(value)


def build_result_metadata(args, artifact_kind, file_name, timestamp=None):
    if artifact_kind not in (SUMMARY_KIND, DETAIL_KIND):
        raise ValueError(f'Unknown evaluation artifact kind: {artifact_kind}')
    if timestamp is None:
        timestamp = current_timestamp()
    sanitized_args = {str(key): _sanitize_metadata_value(value) for key, value in vars(args).items()}
    return {
        'format_version': RESULT_FORMAT_VERSION,
        'timestamp': timestamp,
        'artifact_kind': artifact_kind,
        'file_name': file_name,
        'algorithm': args.algorithm,
        'backbone_source': getattr(args, 'backbone_source', None) or '',
        'dataset': args.dataset,
        'goal': args.goal,
        'seed': int(args.seed),
        'global_rounds': int(args.global_rounds),
        'local_epochs': int(args.local_epochs),
        'args_json': json.dumps(sanitized_args, ensure_ascii=False, sort_keys=True),
    }
