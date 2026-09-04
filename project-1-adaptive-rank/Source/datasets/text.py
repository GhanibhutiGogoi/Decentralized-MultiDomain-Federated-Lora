# text dataset (AG News)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class AGNewsDataset(Dataset):
    """
    AG News topic classification dataset.
    Synthetic data is used only when explicitly requested.
    """
    VOCAB_SIZE = 10000
    MAX_LEN    = 64

    def __init__(self, split="train", synthetic=False):
        super().__init__()
        self.is_synthetic = bool(synthetic)
        if self.is_synthetic:
            rng = np.random.RandomState(42 if split == "train" else 7)
            n   = 5000 if split == "train" else 1000
            self.data = [
                (torch.randint(1, self.VOCAB_SIZE, (self.MAX_LEN,)),
                 rng.randint(0, 4))
                for _ in range(n)
            ]
            return

        try:
            from torchtext.datasets import AG_NEWS
            from torchtext.data.utils import get_tokenizer
            from torchtext.vocab import build_vocab_from_iterator

            tokenizer = get_tokenizer("basic_english")
            raw       = list(AG_NEWS(split=split))
            vocab     = build_vocab_from_iterator(
                (tokenizer(t) for _, t in raw),
                specials=["<pad>", "<unk>"],
                max_tokens=self.VOCAB_SIZE)
            vocab.set_default_index(vocab["<unk>"])

            self.data = []
            for label, text in raw:
                ids  = vocab(tokenizer(text)[:self.MAX_LEN])
                ids += [0] * (self.MAX_LEN - len(ids))
                self.data.append(
                    (torch.tensor(ids, dtype=torch.long), int(label) - 1))
        except (ImportError, OSError, RuntimeError) as e:
            raise RuntimeError(
                "Unable to load real AG News data. Install a working torchtext "
                "package and ensure AG_NEWS is available, or call "
                "get_agnews(synthetic=True) explicitly."
            ) from e

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x, y = self.data[i]
        return x, int(y)


def get_agnews(batch_size=64, synthetic=False):
    """Returns (train_dataset, test_dataset, test_loader) for AG News."""
    train       = AGNewsDataset("train", synthetic=synthetic)
    test        = AGNewsDataset("test", synthetic=synthetic)
    test_loader = DataLoader(test, batch_size=batch_size)
    return train, test, test_loader
