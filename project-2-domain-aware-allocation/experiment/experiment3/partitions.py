"""Experiment 3 partition helpers built on the shared Project 2 partitioner."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable

from framework.partitioning.dirichlet import dirichlet_label_partition_indices


class PartitionValidationError(ValueError):
    """Raised when a generated partition violates conservation invariants."""


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

