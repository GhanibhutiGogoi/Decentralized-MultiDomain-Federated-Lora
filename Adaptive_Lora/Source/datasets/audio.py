# audio dataset (Speech Commands / synthetic)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class AudioDataset(Dataset):
    """
    Google Speech Commands dataset.
    Falls back to synthetic waveform data if torchaudio is unavailable.
    """
    SAMPLE_RATE = 16000
    NUM_CLASSES = 35

    def __init__(self, split="train"):
        super().__init__()
        self._loaded = False
        try:
            import torchaudio
            subset = "training" if split == "train" else "validation"
            ds     = torchaudio.datasets.SPEECHCOMMANDS(
                "./data", download=True, subset=subset)
            all_labels = sorted(
                {ds[i][2] for i in range(min(500, len(ds)))})
            self.label2idx   = {l: i for i, l in enumerate(all_labels)}
            self.NUM_CLASSES = len(self.label2idx)
            self.data        = ds
            self._loaded     = True

        except Exception as e:
            print(f"[Audio] {e}. Using synthetic data.")
            rng = np.random.RandomState(3 if split == "train" else 9)
            n   = 3000 if split == "train" else 600
            self.synth = [
                (torch.from_numpy(
                    rng.randn(1, self.SAMPLE_RATE).astype(np.float32)),
                 rng.randint(0, self.NUM_CLASSES))
                for _ in range(n)
            ]

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


def get_audio(batch_size=32):
    """Returns (train_dataset, test_dataset, test_loader) for audio data."""
    train       = AudioDataset("train")
    test        = AudioDataset("test")
    test_loader = DataLoader(test, batch_size=batch_size)
    return train, test, test_loader
