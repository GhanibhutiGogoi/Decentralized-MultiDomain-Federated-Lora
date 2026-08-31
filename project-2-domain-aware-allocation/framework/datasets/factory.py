"""Centralized dataset loading, validation, and manifest generation.

All Project 2 experiments should obtain datasets through ``DatasetFactory``.
The factory enforces the real-data default, makes download behavior explicit,
and records dataset provenance before training starts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable

from torch.utils.data import DataLoader, Dataset

from framework.partitioning.labels import extract_labels

from .audio import get_audio
from .image import get_cifar10, get_cifar100, get_fashion_mnist
from .tabular import get_tabular
from .text import AGNewsDataset, get_agnews


PROJECT2_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT2_ROOT / "experiment" / "data"


class DatasetValidationError(RuntimeError):
    """Raised when a dataset cannot pass pre-training validation."""


@dataclass(frozen=True)
class DatasetConfig:
    """Standard dataset loading configuration."""

    data_root: Path = DEFAULT_DATA_ROOT
    download: bool = False
    synthetic: bool = False
    batch_size: int | None = None
    num_workers: int = 0
    pin_memory: bool = False
    loader_kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class DatasetSpec:
    """Static metadata and loader for a registered dataset."""

    name: str
    display_name: str
    dataset_type: str
    source_library: str
    source_package: str | None
    version: str | None
    expected_train_samples: int | None
    expected_test_samples: int | None
    expected_classes: int
    default_batch_size: int
    supports_synthetic: bool
    synthetic_train_samples: int | None
    synthetic_test_samples: int | None
    synthetic_classes: int | None
    loader: Callable[..., tuple[Dataset, Dataset, DataLoader]]
    cache_checker: Callable[[Path], bool]
    cache_description: str


@dataclass
class DatasetMetadata:
    """Runtime metadata recorded in dataset manifests."""

    name: str
    display_name: str
    source_library: str
    dataset_type: str
    dataset_version: str | None
    data_root: str
    cache_location: str
    cache_status: str
    synthetic: bool
    train_sample_count: int
    test_sample_count: int
    sample_count: int
    class_count: int
    expected_train_sample_count: int | None
    expected_test_sample_count: int | None
    expected_class_count: int
    download_requested: bool
    download_status: str


@dataclass
class DatasetBundle:
    """Loaded datasets plus validated provenance metadata."""

    train: Dataset
    test: Dataset
    test_loader: DataLoader
    metadata: DatasetMetadata

    def manifest_record(self) -> dict:
        return asdict(self.metadata)


def _package_version(package: str | None) -> str | None:
    if package is None:
        return None
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return None


def _all_exist(root: Path, relative_paths: list[str]) -> bool:
    return all((root / rel).exists() for rel in relative_paths)


def _any_exist(root: Path, relative_paths: list[str]) -> bool:
    return any((root / rel).exists() for rel in relative_paths)


def _cifar10_cache_exists(root: Path) -> bool:
    return _all_exist(
        root,
        [
            "cifar-10-batches-py/data_batch_1",
            "cifar-10-batches-py/test_batch",
        ],
    )


def _cifar100_cache_exists(root: Path) -> bool:
    return _all_exist(
        root,
        [
            "cifar-100-python/train",
            "cifar-100-python/test",
        ],
    )


def _fashion_mnist_cache_exists(root: Path) -> bool:
    return _all_exist(
        root,
        [
            "FashionMNIST/raw/train-images-idx3-ubyte",
            "FashionMNIST/raw/train-labels-idx1-ubyte",
            "FashionMNIST/raw/t10k-images-idx3-ubyte",
            "FashionMNIST/raw/t10k-labels-idx1-ubyte",
        ],
    )


def _ag_news_cache_exists(root: Path) -> bool:
    if _all_exist(
        root,
        [
            "datasets/AG_NEWS/train.csv",
            "datasets/AG_NEWS/test.csv",
        ],
    ):
        return True
    if not root.exists():
        return False
    train_files = list(root.rglob("train.csv"))
    test_files = list(root.rglob("test.csv"))
    return bool(train_files and test_files)


def _heart_cache_exists(root: Path) -> bool:
    return (root / "heart.csv").exists()


def _speech_commands_cache_exists(root: Path) -> bool:
    return _any_exist(
        root,
        [
            "SpeechCommands/speech_commands_v0.02",
            "SpeechCommands/speech_commands_v0.01",
        ],
    )


def _infer_class_count(dataset: Dataset) -> int:
    for attr in ("classes", "label2idx"):
        if hasattr(dataset, attr):
            value = getattr(dataset, attr)
            return len(value)
    for attr in ("num_classes", "NUM_CLASSES"):
        if hasattr(dataset, attr):
            return int(getattr(dataset, attr))
    labels = extract_labels(dataset)
    return int(labels.max()) + 1 if labels.size else 0


def _validate_expected(
    failures: list[str],
    label: str,
    actual: int,
    expected: int | None,
) -> None:
    if expected is not None and actual != expected:
        failures.append(f"{label}: expected {expected}, found {actual}")


class DatasetFactory:
    """Single Project 2 entrypoint for loading experiment datasets."""

    TASK_TO_DATASET = {
        "CIFAR-CNN": "cifar10",
        "Fashion-MLP": "fashion_mnist",
        "AGNews-LSTM": "ag_news",
        "Tabular-MLP": "uci_heart_disease",
        "Audio-1DCNN": "speech_commands",
    }

    REGISTRY = {
        "cifar10": DatasetSpec(
            name="cifar10",
            display_name="CIFAR-10",
            dataset_type="image",
            source_library="torchvision.datasets.CIFAR10",
            source_package="torchvision",
            version="CIFAR-10 python archive",
            expected_train_samples=50000,
            expected_test_samples=10000,
            expected_classes=10,
            default_batch_size=64,
            supports_synthetic=False,
            synthetic_train_samples=None,
            synthetic_test_samples=None,
            synthetic_classes=None,
            loader=get_cifar10,
            cache_checker=_cifar10_cache_exists,
            cache_description="cifar-10-batches-py",
        ),
        "cifar100": DatasetSpec(
            name="cifar100",
            display_name="CIFAR-100",
            dataset_type="image",
            source_library="torchvision.datasets.CIFAR100",
            source_package="torchvision",
            version="CIFAR-100 python archive",
            expected_train_samples=50000,
            expected_test_samples=10000,
            expected_classes=100,
            default_batch_size=64,
            supports_synthetic=False,
            synthetic_train_samples=None,
            synthetic_test_samples=None,
            synthetic_classes=None,
            loader=get_cifar100,
            cache_checker=_cifar100_cache_exists,
            cache_description="cifar-100-python",
        ),
        "fashion_mnist": DatasetSpec(
            name="fashion_mnist",
            display_name="FashionMNIST",
            dataset_type="image",
            source_library="torchvision.datasets.FashionMNIST",
            source_package="torchvision",
            version="FashionMNIST raw IDX archive",
            expected_train_samples=60000,
            expected_test_samples=10000,
            expected_classes=10,
            default_batch_size=64,
            supports_synthetic=False,
            synthetic_train_samples=None,
            synthetic_test_samples=None,
            synthetic_classes=None,
            loader=get_fashion_mnist,
            cache_checker=_fashion_mnist_cache_exists,
            cache_description="FashionMNIST/raw",
        ),
        "ag_news": DatasetSpec(
            name="ag_news",
            display_name="AG News",
            dataset_type="text",
            source_library="torchtext.datasets.AG_NEWS",
            source_package="torchtext",
            version="AG_NEWS torchtext CSV archive",
            expected_train_samples=120000,
            expected_test_samples=7600,
            expected_classes=4,
            default_batch_size=64,
            supports_synthetic=True,
            synthetic_train_samples=5000,
            synthetic_test_samples=1000,
            synthetic_classes=4,
            loader=get_agnews,
            cache_checker=_ag_news_cache_exists,
            cache_description="datasets/AG_NEWS",
        ),
        "uci_heart_disease": DatasetSpec(
            name="uci_heart_disease",
            display_name="UCI Heart Disease",
            dataset_type="tabular",
            source_library="UCI processed.cleveland.data",
            source_package="pandas",
            version="processed.cleveland.data",
            expected_train_samples=237,
            expected_test_samples=60,
            expected_classes=2,
            default_batch_size=64,
            supports_synthetic=True,
            synthetic_train_samples=3200,
            synthetic_test_samples=160,
            synthetic_classes=4,
            loader=get_tabular,
            cache_checker=_heart_cache_exists,
            cache_description="heart.csv",
        ),
        "speech_commands": DatasetSpec(
            name="speech_commands",
            display_name="SpeechCommands",
            dataset_type="audio",
            source_library="torchaudio.datasets.SPEECHCOMMANDS",
            source_package="torchaudio",
            version="speech_commands_v0.02 or cached torchaudio default",
            expected_train_samples=None,
            expected_test_samples=None,
            expected_classes=35,
            default_batch_size=32,
            supports_synthetic=True,
            synthetic_train_samples=3000,
            synthetic_test_samples=600,
            synthetic_classes=35,
            loader=get_audio,
            cache_checker=_speech_commands_cache_exists,
            cache_description="SpeechCommands/speech_commands_v0.02",
        ),
    }

    def __init__(self, config: DatasetConfig | None = None):
        self.config = config or DatasetConfig()

    def load_for_task(self, task_name: str, config: DatasetConfig | None = None) -> DatasetBundle:
        try:
            dataset_name = self.TASK_TO_DATASET[task_name]
        except KeyError as exc:
            raise KeyError(f"No dataset registered for task '{task_name}'.") from exc
        return self.load(dataset_name, config=config)

    def load(self, dataset_name: str, config: DatasetConfig | None = None) -> DatasetBundle:
        if dataset_name not in self.REGISTRY:
            raise KeyError(f"Unknown dataset '{dataset_name}'.")

        cfg = config or self.config
        spec = self.REGISTRY[dataset_name]
        data_root = Path(cfg.data_root).resolve()
        batch_size = cfg.batch_size or spec.default_batch_size
        synthetic = bool(cfg.synthetic)
        download = bool(cfg.download)

        if synthetic and not spec.supports_synthetic:
            raise DatasetValidationError(
                f"{spec.display_name} does not support synthetic mode."
            )

        cache_present_before = spec.cache_checker(data_root)
        if not synthetic and not cache_present_before and not download:
            raise DatasetValidationError(
                f"Missing required real dataset cache for {spec.display_name}: "
                f"{data_root / spec.cache_description}. Synthetic data is never "
                "selected automatically; pass synthetic=True only for explicit "
                "synthetic runs, or rerun with dataset downloads enabled."
            )

        train, test, test_loader = spec.loader(
            data_root=str(data_root),
            batch_size=batch_size,
            synthetic=synthetic,
            download=download,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            **(cfg.loader_kwargs or {}),
        )
        cache_present_after = spec.cache_checker(data_root)
        bundle = DatasetBundle(
            train=train,
            test=test,
            test_loader=test_loader,
            metadata=self._metadata(
                spec=spec,
                data_root=data_root,
                synthetic=synthetic,
                download=download,
                cache_present_before=cache_present_before,
                cache_present_after=cache_present_after,
                train=train,
                test=test,
            ),
        )
        self.validate_bundle(bundle, spec)
        return bundle

    def load_tasks(
        self,
        task_names: list[str],
        synthetic_tasks: set[str] | None = None,
        config: DatasetConfig | None = None,
    ) -> dict[str, DatasetBundle]:
        synthetic_tasks = synthetic_tasks or set()
        unknown_synthetic = synthetic_tasks - set(task_names)
        if unknown_synthetic:
            raise DatasetValidationError(
                f"Synthetic dataset requested for unknown task(s): "
                f"{sorted(unknown_synthetic)}"
            )
        bundles = {}
        for task_name in task_names:
            cfg = config or self.config
            task_config = DatasetConfig(
                data_root=cfg.data_root,
                download=cfg.download,
                synthetic=task_name in synthetic_tasks,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
                loader_kwargs=cfg.loader_kwargs,
            )
            bundles[task_name] = self.load_for_task(task_name, task_config)
        return bundles

    def _metadata(
        self,
        spec: DatasetSpec,
        data_root: Path,
        synthetic: bool,
        download: bool,
        cache_present_before: bool,
        cache_present_after: bool,
        train: Dataset,
        test: Dataset,
    ) -> DatasetMetadata:
        train_count = len(train)
        test_count = len(test)
        expected_train = (
            spec.synthetic_train_samples if synthetic else spec.expected_train_samples
        )
        expected_test = (
            spec.synthetic_test_samples if synthetic else spec.expected_test_samples
        )
        expected_classes = (
            spec.synthetic_classes if synthetic and spec.synthetic_classes else spec.expected_classes
        )
        if synthetic:
            cache_status = "not_applicable"
            download_status = "not_applicable"
        else:
            cache_status = "present" if cache_present_after else "missing"
            if download and not cache_present_before and cache_present_after:
                download_status = "downloaded"
            elif download:
                download_status = "requested_cache_hit"
            else:
                download_status = "disabled_cache_hit"

        return DatasetMetadata(
            name=spec.name,
            display_name=spec.display_name,
            source_library=spec.source_library,
            dataset_type=spec.dataset_type,
            dataset_version=spec.version,
            data_root=str(data_root),
            cache_location=str(data_root / spec.cache_description),
            cache_status=cache_status,
            synthetic=synthetic,
            train_sample_count=train_count,
            test_sample_count=test_count,
            sample_count=train_count + test_count,
            class_count=_infer_class_count(train),
            expected_train_sample_count=expected_train,
            expected_test_sample_count=expected_test,
            expected_class_count=int(expected_classes),
            download_requested=download,
            download_status=download_status,
        )

    def validate_bundle(self, bundle: DatasetBundle, spec: DatasetSpec) -> None:
        metadata = bundle.metadata
        failures: list[str] = []

        if not isinstance(bundle.train, Dataset):
            failures.append("train split is not a torch Dataset")
        if not isinstance(bundle.test, Dataset):
            failures.append("test split is not a torch Dataset")
        if not isinstance(bundle.test_loader, DataLoader):
            failures.append("test_loader is not a torch DataLoader")
        if not metadata.synthetic and metadata.cache_status != "present":
            failures.append("real dataset cache is missing")
        if metadata.synthetic and not spec.supports_synthetic:
            failures.append("synthetic data requested for unsupported dataset")

        _validate_expected(
            failures,
            "train sample count",
            metadata.train_sample_count,
            metadata.expected_train_sample_count,
        )
        _validate_expected(
            failures,
            "test sample count",
            metadata.test_sample_count,
            metadata.expected_test_sample_count,
        )
        _validate_expected(
            failures,
            "class count",
            metadata.class_count,
            metadata.expected_class_count,
        )

        if failures:
            raise DatasetValidationError(
                f"Dataset validation failed for {metadata.display_name}: "
                + "; ".join(failures)
            )


def dataset_manifest_records(bundles: dict[str, DatasetBundle]) -> dict[str, dict]:
    """Return manifest records keyed by experiment task name."""
    return {
        task_name: bundle.manifest_record()
        for task_name, bundle in sorted(bundles.items())
    }


def write_dataset_manifest(
    output_dir: Path,
    experiment_name: str,
    bundles: dict[str, DatasetBundle],
) -> dict:
    """Write and return a dataset manifest for an experiment."""
    data_roots = sorted(
        {
            bundle.metadata.data_root
            for bundle in bundles.values()
            if bundle.metadata.cache_status != "not_applicable"
        }
    )
    manifest = {
        "experiment": experiment_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_roots": data_roots,
        "real_data_default": True,
        "synthetic_requires_explicit_config": True,
        "datasets": dataset_manifest_records(bundles),
        "package_versions": {
            "torch": _package_version("torch"),
            "torchvision": _package_version("torchvision"),
            "torchtext": _package_version("torchtext"),
            "torchaudio": _package_version("torchaudio"),
            "pandas": _package_version("pandas"),
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
            "scikit-learn": _package_version("scikit-learn"),
            "matplotlib": _package_version("matplotlib"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest
