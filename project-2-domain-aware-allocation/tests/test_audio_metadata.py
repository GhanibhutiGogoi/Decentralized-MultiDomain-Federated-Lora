"""SpeechCommands label discovery must not decode the audio corpus."""

import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

PROJECT2_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT2_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT2_ROOT))

from framework.datasets.audio import AudioDataset, get_audio
from framework.partitioning.labels import extract_labels


class _SpeechCommands:
    instances = []

    def __init__(self, root, download, subset):
        self.subset = subset
        self.labels = ["yes", "no", "yes"] if subset == "training" else ["no", "yes"]
        self.decoded_indices = []
        self.metadata_indices = []
        self.instances.append(self)

    def __len__(self):
        return len(self.labels)

    def get_metadata(self, index):
        self.metadata_indices.append(index)
        label = self.labels[index]
        return (f"{label}/speaker_nohash_0.wav", 16000, label, "speaker", 0)

    def __getitem__(self, index):
        self.decoded_indices.append(index)
        return (torch.ones(1, 8000), 16000, self.labels[index], "speaker", 0)


def _mock_torchaudio():
    module = types.ModuleType("torchaudio")
    module.datasets = types.SimpleNamespace(SPEECHCOMMANDS=_SpeechCommands)
    return {"torchaudio": module}


def test_label_mapping_and_partition_labels_use_only_metadata():
    _SpeechCommands.instances = []
    with patch.dict(sys.modules, _mock_torchaudio()):
        train, test, _ = get_audio(data_root="/unused", download=False)
    train_raw, test_raw = _SpeechCommands.instances

    assert train.label2idx == test.label2idx == {"no": 0, "yes": 1}
    np.testing.assert_array_equal(extract_labels(train), [1, 0, 1])
    np.testing.assert_array_equal(extract_labels(test), [0, 1])
    assert train_raw.decoded_indices == test_raw.decoded_indices == []
    assert train_raw.metadata_indices == [0, 1, 2]
    assert test_raw.metadata_indices == [0, 1]

    waveform, label = test[1]
    assert label == 1
    assert waveform.shape == (1, 16000)
    assert test_raw.decoded_indices == [1]
    assert torch.count_nonzero(waveform[:, 8000:]) == 0


def test_unknown_evaluation_labels_are_not_silently_mapped_to_class_zero():
    with patch.dict(sys.modules, _mock_torchaudio()):
        with pytest.raises(ValueError, match="missing from the training mapping"):
            AudioDataset("test", data_root="/unused", label2idx={"yes": 0})
