"""Adapter signatures that retain which output rows and directions changed.

These functions do not accept domain labels. Experiment 05 uses labels only
after clustering to score recovery. All signatures use the scaled effective
update (alpha / rank) B A, so they support heterogeneous ranks and ignore
factor gauge. No privacy guarantee is implied by sharing a signature.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
import torch

from src.federated.merge import lora_to_delta


def _deltas(lora_state, alpha):
    deltas = lora_to_delta(lora_state, alpha)
    return [deltas[name].detach().to(device="cpu", dtype=torch.float64)
            for name in sorted(deltas)]


def signature_row_norms(lora_state, alpha):
    """Concatenated per-output-row L2 norms, normalized to sum to one.

    A zero adapter returns zeros. Layers are sorted by name and may have
    different input/output dimensions; no matrix stacking assumption is made.
    """
    norms = torch.cat([torch.linalg.vector_norm(delta, dim=1)
                       for delta in _deltas(lora_state, alpha)]).numpy()
    total = float(norms.sum())
    return norms / total if total > 0 else norms


def signature_delta_vec(lora_state, alpha):
    """Flatten each scaled delta-W and concatenate in sorted layer order."""
    return torch.cat([delta.reshape(-1) for delta in _deltas(lora_state, alpha)]).numpy()


def affinity_matrix(signatures, kind="cosine"):
    """Pairwise cosine or inverse-L2 similarity with a unit diagonal.

    Inverse-L2 is 1 / (1 + ||a-b||_2), without dataset-dependent rescaling.
    Cosine is defined as one for two zero vectors and zero when only one is
    zero. Its range is [-1, 1]; row-norm signatures have nonnegative cosine.
    """
    if kind not in {"cosine", "inv_l2"}:
        raise ValueError(f"unknown affinity kind: {kind}")
    values = [np.asarray(signature, dtype=np.float64) for signature in signatures]
    if not values or values[0].ndim != 1 or not values[0].size:
        raise ValueError("signatures must contain nonempty one-dimensional vectors")
    if any(value.shape != values[0].shape or not np.isfinite(value).all()
           for value in values):
        raise ValueError("signatures must have matching shapes and finite values")
    result = np.eye(len(values), dtype=np.float64)
    for i, a in enumerate(values):
        for j in range(i + 1, len(values)):
            b = values[j]
            if kind == "inv_l2":
                similarity = 1.0 / (1.0 + float(np.linalg.norm(a - b)))
            else:
                norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
                if norm_a == 0.0 or norm_b == 0.0:
                    similarity = 1.0 if norm_a == norm_b else 0.0
                else:
                    similarity = float(np.clip(np.dot(a / norm_a, b / norm_b), -1.0, 1.0))
            result[i, j] = result[j, i] = similarity
    return result


def cluster_from_affinity(affinity, n_clusters=5):
    """Average-link agglomerative clustering on distance 1 - affinity.

    The number of clusters is a prespecified experimental input. True client
    domain memberships never enter this function.
    """
    values = np.asarray(affinity, dtype=np.float64)
    if (values.ndim != 2 or values.shape[0] != values.shape[1]
            or values.shape[0] < 2):
        raise ValueError("affinity must be a square matrix with at least two clients")
    if (not np.isfinite(values).all() or np.max(np.abs(values - values.T)) > 1e-10
            or np.max(np.abs(np.diag(values) - 1.0)) > 1e-10
            or values.min() < -1.0 - 1e-10 or values.max() > 1.0 + 1e-10):
        raise ValueError("affinity must be finite, symmetric, in [-1, 1] with unit diagonal")
    if not isinstance(n_clusters, (int, np.integer)) or not 1 <= n_clusters <= len(values):
        raise ValueError("n_clusters must be an integer between one and the client count")
    distance = np.maximum(1.0 - values, 0.0)
    np.fill_diagonal(distance, 0.0)
    return AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed",
                                   linkage="average").fit_predict(distance)
