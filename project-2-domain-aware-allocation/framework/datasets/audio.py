"""Audio dataset adapters used by the centralized dataset factory."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "experiment" / "data"


class AudioDataset(Dataset):
    """
    Google Speech Commands dataset.
    Synthetic waveform data is used only when explicitly requested.
    """
    SAMPLE_RATE = 16000
    NUM_CLASSES = 35

    def __init__(
        self,
        split="train",
        data_root=None,
        synthetic=False,
        download=False,
        label2idx=None,
    ):
        super().__init__()
        self.is_synthetic = bool(synthetic)
        data_root = str(data_root or DEFAULT_DATA_ROOT)
        self.data_root = data_root
        self._loaded = False
        if self.is_synthetic:
            rng = np.random.RandomState(3 if split == "train" else 9)
            n = 3000 if split == "train" else 600
            self.synth = [
                (torch.from_numpy(
                    rng.randn(1, self.SAMPLE_RATE).astype(np.float32)),
                 rng.randint(0, self.NUM_CLASSES))
                for _ in range(n)
            ]
            return

        try:
            import torchaudio
            subset = "training" if split == "train" else "validation"
            ds = torchaudio.datasets.SPEECHCOMMANDS(
                data_root, download=download, subset=subset)
            if label2idx is None:
                all_labels = sorted({ds[i][2] for i in range(len(ds))})
                label2idx = {label: i for i, label in enumerate(all_labels)}
            self.label2idx = dict(label2idx)
            self.NUM_CLASSES = len(self.label2idx)
            self.data = ds
            self._loaded = True
        except (ImportError, OSError, RuntimeError) as e:
            raise RuntimeError(
                "Unable to load real SpeechCommands audio data. Install a "
                "working torchaudio package and ensure the dataset is "
                f"available under {data_root}, rerun with dataset downloads "
                "enabled, or call get_audio(synthetic=True) explicitly."
            ) from e

    def __len__(self):
        return len(self.data) if self._loaded else len(self.synth)

    def __getitem__(self, i):
        if self._loaded:
            waveform, sr, label, *_ = self.data[i]
            t = self.SAMPLE_RATE
            waveform = (
                torch.nn.functional.pad(
                    waveform, (0, t - waveform.shape[-1]))
                if waveform.shape[-1] < t
                else waveform[:, :t])
            return waveform, self.label2idx.get(label, 0)
        return self.synth[i]


def get_audio(
    batch_size=32,
    data_root=None,
    synthetic=False,
    download=False,
    num_workers=0,
    pin_memory=False,
):
    """Returns (train_dataset, test_dataset, test_loader) for audio data."""
    train = AudioDataset(
        "train", data_root=data_root, synthetic=synthetic, download=download)
    test = AudioDataset(
        "test",
        data_root=data_root,
        synthetic=synthetic,
        download=download,
        label2idx=getattr(train, "label2idx", None),
    )
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train, test, test_loader
