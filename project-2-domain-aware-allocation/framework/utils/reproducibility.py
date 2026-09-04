"""Reproducibility helpers for Project 2 experiment runners."""

from __future__ import annotations

import os
import platform
import random
import sys
from importlib import metadata as importlib_metadata

import numpy as np
import torch


def set_reproducibility_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch deterministic controls."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _package_version(package: str) -> str | None:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return None


def environment_manifest() -> dict:
    """Return runtime package and platform versions for experiment manifests."""
    packages = [
        "torch",
        "torchvision",
        "torchtext",
        "torchaudio",
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "pyyaml",
        "tqdm",
    ]
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "packages": {
            package: _package_version(package)
            for package in packages
        },
    }
