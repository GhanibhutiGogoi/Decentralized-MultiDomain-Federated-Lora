"""Client partitioning framework."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Subset

from framework.partitioning.dirichlet import dirichlet_label_partition_indices
from framework.partitioning.iid import iid_partition_indices
from framework.partitioning.labels import extract_labels, num_classes_from_labels


@dataclass(frozen=True)
class PartitionConfig:
    """Configuration for client data partitioning."""

    strategy: str = "iid"
    alpha: float = 0.5
    seed: int = 42
    min_client_size: int = 1
    num_workers: int = 0
    pin_memory: bool = False


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
    loaders = []
    for client_id, subset in enumerate(subsets):
        generator = torch.Generator().manual_seed(config.seed + client_id)
        loaders.append(
            DataLoader(
                subset,
                batch_size=batch_sizes[client_id],
                shuffle=True,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                generator=generator,
            )
        )
    return loaders, metadata

