# tabular dataset (Heart Disease / synthetic)

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TabularDataset(Dataset):
    """
    UCI Heart Disease dataset.
    Falls back to synthetic 4-class tabular data if unavailable.
    """
    def __init__(self, split="train"):
        super().__init__()
        try:
            import urllib.request
            import pandas as pd

            url  = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                    "heart-disease/processed.cleveland.data")
            path = "./data/heart.csv"
            os.makedirs("./data", exist_ok=True)
            if not os.path.exists(path):
                urllib.request.urlretrieve(url, path)

            df = pd.read_csv(path, header=None, na_values="?").dropna()
            X  = df.iloc[:, :-1].values.astype(np.float32)
            y  = (df.iloc[:, -1].values > 0).astype(np.int64)

        except Exception as e:
            print(f"[Tabular] {e}. Using synthetic data.")
            rng = np.random.RandomState(0)
            n   = 4000 if split == "train" else 800
            centers = rng.randn(4, 20) * 2
            X = np.vstack(
                [rng.randn(n // 4, 20) + centers[i] for i in range(4)]
            ).astype(np.float32)
            y = np.concatenate(
                [np.full(n // 4, i, dtype=np.int64) for i in range(4)])

        # Normalise
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)

        rng = np.random.RandomState(1)
        idx = rng.permutation(len(X))
        sp  = int(0.8 * len(X))
        idx = idx[:sp] if split == "train" else idx[sp:]

        self.X           = torch.from_numpy(X[idx])
        self.y           = torch.from_numpy(y[idx])
        self.in_dim      = X.shape[1]
        self.num_classes = int(y.max()) + 1

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def get_tabular(batch_size=64):
    """Returns (train_dataset, test_dataset, test_loader) for tabular data."""
    train       = TabularDataset("train")
    test        = TabularDataset("test")
    test_loader = DataLoader(test, batch_size=batch_size)
    return train, test, test_loader
