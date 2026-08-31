"""Tabular dataset adapters used by the centralized dataset factory."""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "experiment" / "data"


class TabularDataset(Dataset):
    """
    UCI Heart Disease dataset.
    Synthetic 4-class tabular data is used only when explicitly requested.
    """
    def __init__(self, split="train", data_root=None, synthetic=False, download=False):
        super().__init__()
        self.is_synthetic = bool(synthetic)
        data_root = str(data_root or DEFAULT_DATA_ROOT)
        self.data_root = data_root
        if self.is_synthetic:
            rng = np.random.RandomState(0)
            n = 4000 if split == "train" else 800
            centers = rng.randn(4, 20) * 2
            X = np.vstack(
                [rng.randn(n // 4, 20) + centers[i] for i in range(4)]
            ).astype(np.float32)
            y = np.concatenate(
                [np.full(n // 4, i, dtype=np.int64) for i in range(4)])
        else:
            try:
                import urllib.error
                import urllib.request
                import pandas as pd

                url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                       "heart-disease/processed.cleveland.data")
                path = os.path.join(data_root, "heart.csv")
                os.makedirs(data_root, exist_ok=True)
                if not os.path.exists(path):
                    if not download:
                        raise FileNotFoundError(path)
                    urllib.request.urlretrieve(url, path)

                df = pd.read_csv(path, header=None, na_values="?").dropna()
                X = df.iloc[:, :-1].values.astype(np.float32)
                y = (df.iloc[:, -1].values > 0).astype(np.int64)
            except (ImportError, OSError, urllib.error.URLError, ValueError) as e:
                raise RuntimeError(
                    "Unable to load real UCI Heart Disease data from "
                    f"{os.path.join(data_root, 'heart.csv')}. Provide the real CSV, "
                    "rerun with dataset downloads enabled, or call "
                    "get_tabular(synthetic=True) explicitly."
                ) from e

        X = (X - X.mean(0)) / (X.std(0) + 1e-8)

        rng = np.random.RandomState(1)
        idx = rng.permutation(len(X))
        sp = int(0.8 * len(X))
        idx = idx[:sp] if split == "train" else idx[sp:]

        self.X = torch.from_numpy(X[idx])
        self.y = torch.from_numpy(y[idx])
        self.in_dim = X.shape[1]
        self.num_classes = int(y.max()) + 1

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def get_tabular(
    batch_size=64,
    data_root=None,
    synthetic=False,
    download=False,
    num_workers=0,
    pin_memory=False,
):
    """Returns (train_dataset, test_dataset, test_loader) for tabular data."""
    train = TabularDataset(
        "train", data_root=data_root, synthetic=synthetic, download=download)
    test = TabularDataset(
        "test", data_root=data_root, synthetic=synthetic, download=download)
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train, test, test_loader
