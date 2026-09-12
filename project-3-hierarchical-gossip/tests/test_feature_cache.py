"""The cache identity must survive JSON storage and reject metadata changes."""

import json

import pytest
import torch

from experiments.feature_cache import cache_identity, load_features, tensor_digest, write_json


def test_cache_identity_roundtrip_and_tampered_weight_provenance(tmp_path):
    cache = tmp_path / "cifar100-resnet18-imagenet1k-v1-resize32-v1"
    cache.mkdir()
    value = {"features": torch.tensor([[1.0, 2.0], [3.0, 4.0]]), "labels": torch.tensor([0, 1])}
    metadata = {"schema_version": 1, "dataset": "CIFAR-100 official train/test split",
        "representation": "frozen ResNet-18, fc replaced by Identity, eval mode",
        "weights": "ResNet18_Weights.IMAGENET1K_V1", "weights_sha256": "verified-checkpoint-hash",
        "transform": {"resize": [32, 32], "mean": [0.485, 0.456, 0.406]},
        "train": {"sha256": tensor_digest(value["features"], value["labels"])},
        "test": {"sha256": tensor_digest(value["features"], value["labels"])}}
    metadata["cache_identity_sha256"] = cache_identity(metadata)
    for split in ("train", "test"):
        torch.save(value, cache / f"{split}.pt")
    write_json(cache / "manifest.json", metadata)
    roundtrip = json.loads((cache / "manifest.json").read_text())
    assert cache_identity(roundtrip) == roundtrip["cache_identity_sha256"]
    train, test, loaded_metadata = load_features(tmp_path / "unused", tmp_path, "cpu", image_size=32)
    assert torch.equal(train["features"], value["features"])
    assert torch.equal(test["labels"], value["labels"])
    assert loaded_metadata["cache_identity_sha256"] == metadata["cache_identity_sha256"]
    roundtrip["weights_sha256"] = "changed-checkpoint"
    write_json(cache / "manifest.json", roundtrip)
    with pytest.raises(ValueError, match="metadata identity mismatch"):
        load_features(tmp_path / "unused", tmp_path, "cpu", image_size=32)


def test_cache_identity_changes_when_transform_changes():
    metadata = {"schema_version": 1, "dataset": "CIFAR-100", "representation": "frozen",
                "weights": "fixed", "weights_sha256": "abc", "transform": {"resize": [32, 32]},
                "train": {"sha256": "def"}, "test": {"sha256": "ghi"}}
    original = cache_identity(metadata)
    metadata["transform"]["resize"] = [224, 224]
    assert cache_identity(metadata) != original
