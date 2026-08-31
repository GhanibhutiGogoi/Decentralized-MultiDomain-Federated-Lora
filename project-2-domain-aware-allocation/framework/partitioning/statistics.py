"""Partition statistics utilities."""

import numpy as np


EPS = 1e-12


def normalized(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    return counts / total if total > 0 else np.zeros_like(counts, dtype=float)


def entropy(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    nz = probs[probs > 0]
    return float(-(nz * np.log(nz)).sum()) if nz.size else 0.0


def class_imbalance_ratio(counts: np.ndarray) -> float:
    """Return a finite imbalance ratio that accounts for absent classes.

    The base ratio is max(counts) / min(counts[counts > 0]). Missing classes
    increase the score through a finite multiplier:

    ``1 + zero_class_count / num_classes``.

    This preserves the signal that absent classes imply stronger imbalance
    while avoiding EPS-driven values disconnected from the observed data.
    Empty clients return 0.0 because no empirical distribution exists.
    """
    counts = np.asarray(counts, dtype=float)
    if counts.size == 0 or counts.sum() <= 0:
        return 0.0
    positive = counts[counts > 0]
    base_ratio = float(counts.max() / positive.min())
    missing_fraction = float(np.sum(counts == 0) / counts.size)
    return base_ratio * (1.0 + missing_fraction)


def client_class_counts(labels, indices_by_client, num_classes: int):
    labels = np.asarray(labels, dtype=np.int64)
    return [
        np.bincount(labels[np.asarray(indices, dtype=np.int64)], minlength=num_classes)
        for indices in indices_by_client
    ]
