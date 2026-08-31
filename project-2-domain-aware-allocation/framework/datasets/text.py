"""Text dataset adapters used by the centralized dataset factory."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "experiment" / "data"


class AGNewsDataset(Dataset):
    """
    AG News topic classification dataset.
    Synthetic data is used only when explicitly requested.
    """
    VOCAB_SIZE = 10000
    MAX_LEN = 64

    def __init__(self, split="train", data_root=None, synthetic=False, download=False):
        super().__init__()
        self.is_synthetic = bool(synthetic)
        data_root = str(data_root or DEFAULT_DATA_ROOT)
        self.data_root = str(data_root)
        if self.is_synthetic:
            rng = np.random.RandomState(42 if split == "train" else 7)
            generator = torch.Generator().manual_seed(42 if split == "train" else 7)
            n = 5000 if split == "train" else 1000
            self.data = [
                (torch.randint(1, self.VOCAB_SIZE, (self.MAX_LEN,), generator=generator),
                 rng.randint(0, 4))
                for _ in range(n)
            ]
            return

        try:
            from torchtext.datasets import AG_NEWS
            from torchtext.data.utils import get_tokenizer
            from torchtext.vocab import build_vocab_from_iterator

            cache_root = Path(data_root)
            split_file = cache_root / "datasets" / "AG_NEWS" / f"{split}.csv"
            if not download and not split_file.exists():
                matches = list(cache_root.rglob(f"{split}.csv")) if cache_root.exists() else []
                if not matches:
                    raise FileNotFoundError(split_file)

            tokenizer = get_tokenizer("basic_english")
            raw = list(AG_NEWS(root=str(data_root), split=split))
            vocab = build_vocab_from_iterator(
                (tokenizer(t) for _, t in raw),
                specials=["<pad>", "<unk>"],
                max_tokens=self.VOCAB_SIZE)
            vocab.set_default_index(vocab["<unk>"])

            self.data = []
            for label, text in raw:
                ids = vocab(tokenizer(text)[:self.MAX_LEN])
                ids += [0] * (self.MAX_LEN - len(ids))
                self.data.append(
                    (torch.tensor(ids, dtype=torch.long), int(label) - 1))
        except (ImportError, OSError, RuntimeError, FileNotFoundError) as e:
            raise RuntimeError(
                "Unable to load real AG News data. Install a working torchtext "
                f"package and ensure AG_NEWS is cached under {data_root}, rerun "
                "with dataset downloads enabled, or call get_agnews(synthetic=True) "
                "explicitly."
            ) from e

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x, y = self.data[i]
        return x, int(y)


def get_agnews(
    batch_size=64,
    data_root=None,
    synthetic=False,
    download=False,
    num_workers=0,
    pin_memory=False,
):
    """Returns (train_dataset, test_dataset, test_loader) for AG News."""
    train = AGNewsDataset(
        "train", data_root=data_root, synthetic=synthetic, download=download)
    test = AGNewsDataset(
        "test", data_root=data_root, synthetic=synthetic, download=download)
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train, test, test_loader
