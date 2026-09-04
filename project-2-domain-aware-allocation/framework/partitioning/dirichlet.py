"""Dirichlet label partitioning."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from framework.partitioning.labels import _to_numpy_labels


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

