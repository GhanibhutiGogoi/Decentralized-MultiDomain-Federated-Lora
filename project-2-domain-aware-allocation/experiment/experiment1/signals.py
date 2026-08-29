"""Domain signal measurement utilities for Experiment 1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


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

    The input is the complete class-count vector for a client. The ratio is
    max(counts) / max(min(counts), EPS), so zero-count classes contribute
    through the EPS denominator floor. Empty clients return 0.0 because no
    empirical class distribution exists.
    """
    counts = np.asarray(counts, dtype=float)
    if counts.size == 0 or counts.sum() <= 0:
        return 0.0
    return float(counts.max() / max(float(counts.min()), EPS))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float) + EPS
    q = np.asarray(q, dtype=float) + EPS
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float) + EPS
    q = np.asarray(q, dtype=float) + EPS
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m))


def label_distribution_records(
    task_name: str,
    labels: np.ndarray,
    indices_by_client: list[list[int]],
    num_classes: int,
    is_synthetic: bool = False,
):
    """Build per-client label distribution records and raw count matrix."""
    labels = np.asarray(labels, dtype=np.int64)
    global_counts = np.bincount(labels, minlength=num_classes)
    global_freq = normalized(global_counts)

    records = []
    raw_counts = []
    normalized_freqs = []

    for client_id, indices in enumerate(indices_by_client):
        client_labels = labels[np.asarray(indices, dtype=np.int64)]
        counts = np.bincount(client_labels, minlength=num_classes)
        probs = normalized(counts)
        raw_counts.append(counts)
        normalized_freqs.append(probs)
        records.append(
            {
                "task": task_name,
                "client_id": client_id,
                "is_synthetic": bool(is_synthetic),
                "num_samples": int(counts.sum()),
                "raw_class_counts": counts.astype(int).tolist(),
                "class_frequency": probs.tolist(),
                "entropy": entropy(probs),
                "normalized_entropy": (
                    entropy(probs) / np.log(num_classes) if num_classes > 1 else 0.0
                ),
                "class_imbalance_ratio": class_imbalance_ratio(counts),
                "kl_to_global": kl_divergence(probs, global_freq),
                "js_to_global": js_divergence(probs, global_freq),
                "zero_class_count": int(np.sum(counts == 0)),
            }
        )

    payload = {
        "task": task_name,
        "is_synthetic": bool(is_synthetic),
        "num_classes": int(num_classes),
        "global_class_counts": global_counts.astype(int).tolist(),
        "global_class_frequency": global_freq.tolist(),
        "client_class_counts": np.asarray(raw_counts, dtype=int).tolist(),
        "client_class_frequency": np.asarray(normalized_freqs, dtype=float).tolist(),
    }
    return records, payload


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= EPS:
        return 0.0
    return float(1.0 - np.dot(a, b) / denom)


def update_dissimilarity_records(task_name: str, round_id: int, update_vectors):
    """Measure each client update against the mean client update."""
    if not update_vectors:
        return []

    max_len = max(vec.size for vec in update_vectors)
    padded = []
    for vec in update_vectors:
        if vec.size < max_len:
            vec = np.pad(vec, (0, max_len - vec.size))
        padded.append(vec.astype(float))

    matrix = np.vstack(padded)
    mean_update = matrix.mean(axis=0)

    records = []
    for client_id, vec in enumerate(matrix):
        diff = vec - mean_update
        records.append(
            {
                "task": task_name,
                "round": round_id,
                "client_id": client_id,
                "update_cosine_distance_to_mean": cosine_distance(vec, mean_update),
                "update_l2_distance_to_mean": float(np.linalg.norm(diff)),
                "update_norm": float(np.linalg.norm(vec)),
            }
        )
    return records


def save_label_distribution_outputs(
    records,
    payloads,
    output_dir: Path,
) -> pd.DataFrame:
    """Save CSV summaries and raw machine-readable frequency vectors."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(output_dir / "label_distribution_summary.csv", index=False)

    with (output_dir / "label_distribution_raw.json").open("w", encoding="utf-8") as f:
        json.dump(payloads, f, indent=2)

    return df
