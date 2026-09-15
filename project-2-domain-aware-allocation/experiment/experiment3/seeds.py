"""Deterministic seed bookkeeping for Experiment 3."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Experiment3Seeds:
    """All seeds that must be stored in an Experiment 3 run record."""

    run_seed: int
    partition_seed: int
    model_seed: int
    dataloader_seed: int
    permutation_seed: int

    def to_manifest(self) -> dict[str, int]:
        return asdict(self)


def derive_seeds(run_seed: int) -> Experiment3Seeds:
    """Derive deterministic, separated seed streams from one run seed."""
    base = int(run_seed)
    return Experiment3Seeds(
        run_seed=base,
        partition_seed=base + 1009,
        model_seed=base + 2003,
        dataloader_seed=base + 3001,
        permutation_seed=base + 4001,
    )

