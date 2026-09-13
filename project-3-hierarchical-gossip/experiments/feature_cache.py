"""Deterministic, shared frozen ResNet-18 features for the completion benchmark.

This is explicitly a frozen-feature classification experiment. It does not
replicate the augmentation or mutable BatchNorm protocol of experiments 01–03.
"""

import hashlib
import json
import os
from pathlib import Path
import time

import torch


def tensor_digest(*tensors):
    digest = hashlib.sha256()
    for value in tensors:
        value = value.detach().cpu().contiguous()
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def cache_identity(metadata):
    """Bind the representation protocol and source weights to tensor hashes."""
    identity = {key: metadata[key] for key in
                ("schema_version", "dataset", "representation", "weights", "weights_sha256", "transform")}
    identity["feature_sha256"] = {split: metadata[split]["sha256"] for split in ("train", "test")}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    """Replace a JSON file atomically; a reader never observes half a record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_features(data_dir, cache_dir, device, image_size=224, batch_size=256,
                  num_workers=2):
    """Return train/test CPU feature tensors, labels and cache provenance.

    Cache names include every transform/model choice. Tensor digests are checked
    on reuse, and the manifest is written only after both feature files exist.
    Torchvision checks the original CIFAR-100 archive's MD5 when extracting it.
    """
    import torchvision
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms

    if image_size < 32:
        raise ValueError("image_size must be >= 32")
    cache = Path(cache_dir) / f"cifar100-resnet18-imagenet1k-v1-resize{image_size}-v1"
    metadata_path = cache / "manifest.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("cache_identity_sha256") != cache_identity(metadata):
            raise ValueError(f"feature cache metadata identity mismatch: {metadata_path}")
        if metadata["transform"]["resize"] != [image_size, image_size] or metadata["weights"] != "ResNet18_Weights.IMAGENET1K_V1":
            raise ValueError(f"feature cache protocol mismatch: {metadata_path}")
        values = [torch.load(cache / f"{split}.pt", map_location="cpu", weights_only=True)
                  for split in ("train", "test")]
        for split, value in zip(("train", "test"), values):
            if tensor_digest(value["features"], value["labels"]) != metadata[split]["sha256"]:
                raise ValueError(f"feature cache digest mismatch: {cache / (split + '.pt')}")
        return values[0], values[1], metadata

    started = time.perf_counter()
    cache.mkdir(parents=True, exist_ok=True)
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    backbone = models.resnet18(weights=weights)
    backbone.fc = torch.nn.Identity()
    backbone = backbone.to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    metadata = {
        "schema_version": 1,
        "dataset": "CIFAR-100 official train/test split",
        "representation": "frozen ResNet-18, fc replaced by Identity, eval mode",
        "weights": "ResNet18_Weights.IMAGENET1K_V1",
        "weights_url": weights.url,
        "transform": {"resize": [image_size, image_size], "interpolation": "bilinear",
                      "antialias": True, "mean": [0.485, 0.456, 0.406],
                      "std": [0.229, 0.224, 0.225], "augmentation": "none"},
        "torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
        "cache_path": str(cache.resolve()),
    }
    result = []
    for split in ("train", "test"):
        dataset = datasets.CIFAR100(root=str(data_dir), train=split == "train",
                                    download=True, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=str(device).startswith("cuda"))
        features, labels = [], []
        with torch.inference_mode():
            for index, (images, target) in enumerate(loader):
                features.append(backbone(images.to(device, non_blocking=True)).cpu())
                labels.append(target)
                if index % 25 == 0:
                    print(f"feature cache {split}: batch {index + 1}/{len(loader)}", flush=True)
        value = {"features": torch.cat(features), "labels": torch.cat(labels)}
        if not torch.isfinite(value["features"]).all():
            raise ValueError("non-finite extracted features")
        temporary = cache / f"{split}.pt.tmp"
        torch.save(value, temporary)
        os.replace(temporary, cache / f"{split}.pt")
        metadata[split] = {"n_samples": len(dataset), "feature_dim": value["features"].shape[1],
                           "sha256": tensor_digest(value["features"], value["labels"])}
        result.append(value)
    weight_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    if not weight_path.exists():
        raise FileNotFoundError(f"pretrained checkpoint unavailable for provenance hash: {weight_path}")
    metadata["weights_sha256"] = hashlib.sha256(weight_path.read_bytes()).hexdigest()
    metadata["cache_identity_sha256"] = cache_identity(metadata)
    metadata["extraction_wall_seconds"] = time.perf_counter() - started
    write_json(metadata_path, metadata)
    del backbone
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return result[0], result[1], metadata
