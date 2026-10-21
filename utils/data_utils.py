import os
from pathlib import Path

import numpy as np
import torch


def read_data(dataset, idx, is_train=True):
    split = "train" if is_train else "test"
    file_path = str(data_root() / dataset / split / f"{idx}.npz")
    with open(file_path, "rb") as f:
        return np.load(f, allow_pickle=True)["data"].tolist()


def read_client_data(dataset, idx, is_train=True):
    if dataset == "FedRS" and not (data_root() / dataset / "train" / "0.npz").exists():
        from utils.fedrs_data import read_fedrs
        data_list = read_fedrs(data_root() / dataset, idx, is_train)
    else:
        data_list = process_image(read_data(dataset, idx, is_train))


    return data_list


def read_global_test_data(dataset):
    if dataset == "FedRS" and not (data_root() / dataset / "global_test.npz").exists():
        from utils.fedrs_data import read_fedrs
        return read_fedrs(data_root() / dataset)
    file_path = str(data_root() / dataset / "global_test.npz")
    if not os.path.exists(file_path):
        return None

    with open(file_path, "rb") as f:
        data = np.load(f, allow_pickle=True)["data"].tolist()
    return process_image(data)


def process_image(data):
    x = torch.tensor(data["x"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.int64)
    return [(image, label) for image, label in zip(x, y)]


def data_root():
    return Path(os.environ.get('FEDGRC_DATA_ROOT', Path(__file__).resolve().parents[1] / 'dataset')).resolve()
