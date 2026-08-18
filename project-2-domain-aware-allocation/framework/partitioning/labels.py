"""Dataset label extraction utilities."""

from __future__ import annotations

import numpy as np


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

