"""Centralized dataset utilities and loaders."""

from .audio import AudioDataset, get_audio
from .factory import (
    DEFAULT_DATA_ROOT,
    DatasetBundle,
    DatasetConfig,
    DatasetFactory,
    DatasetMetadata,
    DatasetValidationError,
    dataset_manifest_records,
    write_dataset_manifest,
)
from .image import get_cifar10, get_cifar100, get_fashion_mnist
from .tabular import TabularDataset, get_tabular
from .text import AGNewsDataset, get_agnews

__all__ = [
    "AGNewsDataset",
    "AudioDataset",
    "DEFAULT_DATA_ROOT",
    "DatasetBundle",
    "DatasetConfig",
    "DatasetFactory",
    "DatasetMetadata",
    "DatasetValidationError",
    "TabularDataset",
    "dataset_manifest_records",
    "get_agnews",
    "get_audio",
    "get_cifar10",
    "get_cifar100",
    "get_fashion_mnist",
    "get_tabular",
    "write_dataset_manifest",
]
