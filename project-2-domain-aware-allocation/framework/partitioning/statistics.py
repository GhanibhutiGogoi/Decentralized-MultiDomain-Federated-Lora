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
    """Return max class count divided by min class count over all classes.

    Zero-count classes are part of the complete class vector. The ratio is
    max(counts) / max(min(counts), EPS), so absent classes contribute through
    the EPS denominator floor. Empty clients return 0.0 because no empirical
    distribution exists.
    """
    counts = np.asarray(counts, dtype=float)
    if counts.size == 0 or counts.sum() <= 0:
        return 0.0
    return float(counts.max() / max(float(counts.min()), EPS))


def client_class_counts(labels, indices_by_client, num_classes: int):
    labels = np.asarray(labels, dtype=np.int64)
    return [
        np.bincount(labels[np.asarray(indices, dtype=np.int64)], minlength=num_classes)
        for indices in indices_by_client
    ]
