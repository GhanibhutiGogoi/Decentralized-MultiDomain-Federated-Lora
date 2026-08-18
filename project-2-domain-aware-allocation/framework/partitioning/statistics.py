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
    counts = np.asarray(counts, dtype=float)
    nonzero = counts[counts > 0]
    if nonzero.size == 0:
        return 0.0
    return float(nonzero.max() / max(nonzero.min(), EPS))


def client_class_counts(labels, indices_by_client, num_classes: int):
    labels = np.asarray(labels, dtype=np.int64)
    return [
        np.bincount(labels[np.asarray(indices, dtype=np.int64)], minlength=num_classes)
        for indices in indices_by_client
    ]

