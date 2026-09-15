"""Experiment 3 partition helpers built on the shared Project 2 partitioner."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from framework.partitioning.dirichlet import dirichlet_label_partition_indices


class PartitionValidationError(ValueError):
    """Raised when a generated partition violates conservation invariants."""


def heldout_domain_indices(test_labels, train_labels, train_indices_by_client, *, seed):
    """Partition held-out examples by training-derived client/class proportions.

    For each class, largest-remainder allocation preserves its complete test
    count while approximating the proportions of that class in client training
    shards. A separate RNG shuffles held-out indices only. These are simulated
    client domains, not externally supplied clinical domain labels.
    """
    train_labels = np.asarray(train_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)
    n_clients = len(train_indices_by_client)
    if not n_clients or not len(test_labels):
        raise PartitionValidationError("Held-out domain evaluation requires clients and test examples.")
    result = [[] for _ in range(n_clients)]
    rng = np.random.default_rng(seed)
    for label in np.unique(test_labels):
        indices = np.flatnonzero(test_labels == label)
        rng.shuffle(indices)
        counts = np.asarray([np.count_nonzero(train_labels[client] == label)
                             for client in train_indices_by_client], dtype=float)
        proportions = counts / counts.sum() if counts.sum() else np.full(n_clients, 1 / n_clients)
        expected = len(indices) * proportions
        sizes = np.floor(expected).astype(int)
        remainder = len(indices) - int(sizes.sum())
        order = np.argsort(-(expected - sizes), kind="stable")
        sizes[order[:remainder]] += 1
        for client, shard in enumerate(np.split(indices, np.cumsum(sizes)[:-1])):
            result[client].extend(shard.tolist())
    validate_partition(result, len(test_labels))
    if any(not shard for shard in result):
        raise PartitionValidationError("A client has no held-out test examples; domain accuracy is undefined.")
    return result


@dataclass(frozen=True)
class PartitionRecord:
    """Validated fresh partition identity for one Experiment 3 task."""

    strategy: str
    alpha: float
    seed: int
    client_ids: tuple[str, ...]
    indices_by_client: tuple[tuple[int, ...], ...]
    partition_hash: str


def _partition_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_partition(indices_by_client: Iterable[Iterable[int]], total_samples: int) -> None:
    """Require every sample index exactly once and no unknown indices."""
    flattened: list[int] = []
    for indices in indices_by_client:
        flattened.extend(int(index) for index in indices)
    expected = set(range(int(total_samples)))
    actual = set(flattened)
    if len(flattened) != len(actual):
        raise PartitionValidationError("Partition contains duplicated sample indices.")
    if actual != expected:
        raise PartitionValidationError("Partition lost or added sample indices.")


def fresh_dirichlet_partition(
    labels: Iterable[int],
    *,
    num_clients: int,
    alpha: float,
    seed: int,
    client_id_prefix: str = "client",
) -> PartitionRecord:
    """Generate a fresh deterministic Dirichlet partition from labels only."""
    labels_list = list(labels)
    indices = dirichlet_label_partition_indices(
        labels=labels_list,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
        min_client_size=1,
    )
    validate_partition(indices, len(labels_list))
    normalized = tuple(tuple(int(index) for index in client) for client in indices)
    client_ids = tuple(f"{client_id_prefix}_{idx}" for idx in range(num_clients))
    return PartitionRecord(
        strategy="dirichlet",
        alpha=float(alpha),
        seed=int(seed),
        client_ids=client_ids,
        indices_by_client=normalized,
        partition_hash=_partition_hash(
            {
                "strategy": "dirichlet",
                "alpha": float(alpha),
                "seed": int(seed),
                "client_ids": client_ids,
                "indices_by_client": normalized,
            }
        ),
    )
