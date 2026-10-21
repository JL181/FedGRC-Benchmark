"""Convert the supplied FedRS NIID-1 images into the benchmark NPZ format.

Uses the same reader as training: 135 clients, 15 semantic classes, fixed
per-class 75/25 views (partition seed 0), Resize(64,64), and ImageNet
normalization.
Existing prepared data are never overwritten.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from utils.fedrs_data import N_CLASSES, read_fedrs


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src", type=Path, default=ROOT / "dataset" / "FedRS",
        help="Image root containing NIID-1/ and test_balanced/",
    )
    parser.add_argument(
        "--dst", type=Path, default=ROOT / "dataset" / "FedRS",
        help="Destination for train/, test/, global_test.npz and config.json",
    )
    return parser


def write_split(path, samples):
    if not samples:
        raise ValueError(f"Empty FedRS split: {path}")
    x = np.asarray([image.numpy() for image, _ in samples], dtype=np.float32)
    y = np.asarray([int(label) for _, label in samples], dtype=np.int64)
    if x.shape != (len(y), 3, 64, 64) or not np.isfinite(x).all():
        raise ValueError(f"Invalid FedRS image tensors: {path}")
    if np.any((y < 0) | (y >= N_CLASSES)):
        raise ValueError(f"Invalid FedRS labels: {path}")
    with path.open("xb") as output:
        np.savez_compressed(output, data={"x": x, "y": y})
    return np.bincount(y, minlength=N_CLASSES), len(y)


def convert(src, dst):
    src, dst = src.resolve(), dst.resolve()
    for directory in [src / "test_balanced"] + [
        src / "NIID-1" / f"client_{cid}" for cid in range(135)
    ]:
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing official FedRS input: {directory}")
    outputs = [dst / name for name in ("train", "test", "global_test.npz", "config.json")]
    occupied = [str(path) for path in outputs if path.exists()]
    if occupied:
        raise FileExistsError(
            "Prepared data already exist; choose a new --dst: " + ", ".join(occupied)
        )
    (dst / "train").mkdir(parents=True, exist_ok=False)
    (dst / "test").mkdir(exist_ok=False)
    statistics, train_sizes, test_sizes = [], [], []
    for cid in range(135):
        train_counts, train_size = write_split(
            dst / "train" / f"{cid}.npz", read_fedrs(src, cid, is_train=True)
        )
        test_counts, test_size = write_split(
            dst / "test" / f"{cid}.npz", read_fedrs(src, cid, is_train=False)
        )
        counts = train_counts + test_counts
        statistics.append([[int(c), int(counts[c])] for c in range(N_CLASSES) if counts[c]])
        train_sizes.append(train_size)
        test_sizes.append(test_size)
        print(f"client_{cid}: train={train_size}, test={test_size}", flush=True)
    global_counts, global_size = write_split(dst / "global_test.npz", read_fedrs(src))
    if np.any(global_counts == 0):
        raise ValueError("The global test set must cover every FedRS class")
    config = {
        "num_clients": 135,
        "num_classes": N_CLASSES,
        "non_iid": True,
        "balance": False,
        "partition": "dir",
        "Size of samples for labels in clients": statistics,
        "alpha": 0.5,
        "min_class_samples_per_client": None,
        "batch_size": 10,
        "train_ratio": 0.75,
        "client_data_ratio": 1.0,
        "global_test_ratio": 0.0,
        "train_samples_per_client": train_sizes,
        "test_samples_per_client": test_sizes,
        "split_tag": "FedRS_NIID-1",
        "source": "FedRS-Bench (arXiv 2505.08325); official imagefolder",
        "normalize": "ImageNet mean/std (official)",
        "image_size": 64,
        "global_test_samples": global_size,
        "global_test_label_counts": global_counts.tolist(),
        "partition_seed": 0,
    }
    with (dst / "config.json").open("x") as output:
        json.dump(config, output, indent=2)
        output.write("\n")
    print(f"Saved 271 NPZ files and config.json to {dst}")


if __name__ == "__main__":
    args = build_parser().parse_args()
    convert(args.src, args.dst)
