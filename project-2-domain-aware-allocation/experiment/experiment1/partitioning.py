"""Client partitioning for Project 2 Experiment 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from torch.utils.data import DataLoader, Subset


@dataclass(frozen=True)
class PartitionConfig:
    """Configuration for client data partitioning."""

    strategy: str = "iid"
    alpha: float = 0.5
    seed: int = 42
    min_client_size: int = 1
    num_workers: int = 0
    pin_memory: bool = False


def _to_numpy_labels(values) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.int64).reshape(-1)


def extract_labels(dataset) -> np.ndarray:
    """Return integer labels for any Project 1 classification dataset."""
    for attr in ("targets", "labels", "y"):
        if hasattr(dataset, attr):
            return _to_numpy_labels(getattr(dataset, attr))

    if hasattr(dataset, "synth"):
        return _to_numpy_labels([int(item[1]) for item in dataset.synth])

    data = getattr(dataset, "data", None)
    if isinstance(data, list) and data and isinstance(data[0], tuple):
        if len(data[0]) >= 2:
            return _to_numpy_labels([int(item[1]) for item in data])

    labels = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise ValueError(
                "Dataset items must return (features, label) to extract labels."
            )
        label = item[1]
        if hasattr(label, "item"):
            label = label.item()
        labels.append(int(label))
    return _to_numpy_labels(labels)


def num_classes_from_labels(labels: np.ndarray) -> int:
    """Infer class count while preserving zero-based class vector semantics."""
    labels = _to_numpy_labels(labels)
    return int(labels.max()) + 1 if labels.size else 0


def iid_partition_indices(dataset, num_clients: int) -> list[list[int]]:
    """Project 1-compatible contiguous split.

    This intentionally drops the final remainder, matching Project 1's
    ``Federated.utilities.split_dataset`` behavior.
    """
    n = len(dataset)
    size = n // num_clients
    return [
        list(range(client_id * size, (client_id + 1) * size))
        for client_id in range(num_clients)
    ]


def dirichlet_label_partition_indices(
    labels: Iterable[int],
    num_clients: int,
    alpha: float,
    seed: int,
    min_client_size: int = 1,
    max_attempts: int = 100,
) -> list[list[int]]:
    """Partition indices by class using Dirichlet label proportions."""
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive.")

    labels = _to_numpy_labels(labels)
    if labels.size == 0:
        raise ValueError("Cannot partition an empty dataset.")

    classes = np.unique(labels)
    if len(labels) < num_clients * min_client_size:
        raise ValueError(
            "Dataset is too small for the requested number of clients and "
            "minimum client size."
        )

    for attempt in range(max_attempts):
        rng = np.random.default_rng(seed + attempt)
        client_indices = [[] for _ in range(num_clients)]

        for cls in classes:
            cls_indices = np.where(labels == cls)[0]
            rng.shuffle(cls_indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=float))
            split_points = (np.cumsum(proportions) * len(cls_indices)).astype(int)
            split_points[-1] = len(cls_indices)
            splits = np.split(cls_indices, split_points[:-1])
            for client_id, split in enumerate(splits):
                client_indices[client_id].extend(split.tolist())

        for indices in client_indices:
            rng.shuffle(indices)

        if min(len(indices) for indices in client_indices) >= min_client_size:
            return client_indices

    raise RuntimeError(
        "Unable to create a non-empty Dirichlet partition. Increase alpha, "
        "reduce client count, or lower min_client_size."
    )


def partition_indices(dataset, num_clients: int, config: PartitionConfig):
    """Return ``(indices_by_client, labels, num_classes)``."""
    strategy = config.strategy.lower()
    labels = extract_labels(dataset)
    num_classes = num_classes_from_labels(labels)

    if strategy in {"iid", "legacy_iid", "project1_iid"}:
        indices = iid_partition_indices(dataset, num_clients)
    elif strategy in {"dirichlet", "dirichlet_label", "label_dirichlet"}:
        indices = dirichlet_label_partition_indices(
            labels=labels,
            num_clients=num_clients,
            alpha=config.alpha,
            seed=config.seed,
            min_client_size=config.min_client_size,
        )
    else:
        raise ValueError(f"Unknown partition strategy: {config.strategy}")

    return indices, labels, num_classes


def make_client_subsets(dataset, num_clients: int, config: PartitionConfig):
    """Create client ``Subset`` objects and partition metadata."""
    indices, labels, num_classes = partition_indices(dataset, num_clients, config)
    subsets = [Subset(dataset, client_indices) for client_indices in indices]
    metadata = {
        "strategy": config.strategy,
        "alpha": config.alpha,
        "seed": config.seed,
        "indices_by_client": indices,
        "labels": labels,
        "num_classes": num_classes,
    }
    return subsets, metadata


def make_client_loaders(dataset, batch_sizes, config: PartitionConfig):
    """Create DataLoaders while keeping hardware batch sizes independent."""
    subsets, metadata = make_client_subsets(dataset, len(batch_sizes), config)
    loaders = [
        DataLoader(
            subset,
            batch_size=batch_sizes[client_id],
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        for client_id, subset in enumerate(subsets)
    ]
    return loaders, metadata

