"""P2 domain allocation and a reversible, weighted gossip kernel.

Weights change the stationary distribution of the proposal, rather than being
erased by doubly-stochastic normalization. The resulting kernel is generally
neither symmetric nor doubly stochastic. Domain features use training labels
and local effective-update changes; sharing these signals is metadata exchange,
not a privacy guarantee. Communication belongs to the caller, which must obtain
the inputs through its declared protocol before invoking these pure functions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import importlib.util
from pathlib import Path

import numpy as np

from .merge import lora_to_delta


_REPO_ROOT = Path(__file__).resolve().parents[3]
_POLICY_PATH = _REPO_ROOT / "project-2-domain-aware-allocation/framework/aggregation/domain_weighting.py"
_SIGNALS_PATH = _REPO_ROOT / "project-2-domain-aware-allocation/experiment/experiment1/signals.py"
_spec = importlib.util.spec_from_file_location("_p2_conservative_domain_policy", _POLICY_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load canonical P2 domain policy: {_POLICY_PATH}")
_policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_policy)
# This is the actual P2 function, with no copied implementation or altered defaults.
conservative_domain_factors = _policy.conservative_domain_factors

DOMAIN_POLICY_PROVENANCE = {
    "implementation": "direct import of canonical P2 conservative_domain_factors",
    "source": str(_POLICY_PATH.relative_to(_REPO_ROOT)),
    "source_sha256": hashlib.sha256(_POLICY_PATH.read_bytes()).hexdigest(),
    "feature_source": str(_SIGNALS_PATH.relative_to(_REPO_ROOT)),
    "feature_source_sha256": hashlib.sha256(_SIGNALS_PATH.read_bytes()).hexdigest(),
    "blend_strength": 0.10,
    "max_deviation": 0.15,
    "temperature": 1.0,
    "feature_names": ["js_to_global", "update_l2_distance_to_mean",
                      "update_cosine_distance_to_mean", "normalized_entropy",
                      "class_imbalance_ratio"],
    "update_space": "effective LoRA delta-W, including alpha/r; local change from pretraining state when supplied",
    "feature_mean": "unweighted arithmetic mean of local updates, as in P2",
    "label_reference": "pooled training histogram, formed by summing client counts",
    "privacy_note": "training histograms and update signals are exchanged metadata; no leakage protection is claimed",
}


def weighted_metropolis(proposal, weights) -> np.ndarray:
    """Turn symmetric stochastic ``proposal`` into a kernel targeting weights.

    For i != j, P[i,j] = W[i,j] min(1, weights[j]/weights[i]); diagonal
    entries retain the rejected mass. Positive finite weights are required.
    Thus pi_i proportional to weights_i satisfies detailed balance and
    pi.T @ P == pi.T. Only existing off-diagonal proposal edges are used.
    Scaling every weight by the same positive constant has no effect.
    """
    w = np.asarray(proposal, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1] or w.shape[0] == 0:
        raise ValueError("proposal must be a nonempty square matrix")
    if not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("proposal must have finite non-negative entries")
    if not np.allclose(w, w.T, atol=1e-12, rtol=0):
        raise ValueError("proposal must be symmetric")
    if not np.allclose(w.sum(axis=1), 1.0, atol=1e-12, rtol=0):
        raise ValueError("proposal must be row-stochastic")
    target = np.asarray(weights, dtype=float)
    if target.shape != (len(w),) or not np.isfinite(target).all() or np.any(target <= 0):
        raise ValueError("weights must contain one finite strictly positive value per peer")
    # Avoid overflow in the sum or ratios; computing smaller/larger gives
    # numbers in [0, 1], even for large finite differences in scale.
    numerator = np.minimum(target[:, None], target[None, :])
    acceptance = numerator / target[:, None]
    p = w * acceptance
    np.fill_diagonal(p, 0.0)
    diagonal = 1.0 - p.sum(axis=1)
    if np.any(diagonal < -1e-12):
        raise ValueError("proposal produces negative diagonal mass")
    np.fill_diagonal(p, np.maximum(diagonal, 0.0))
    return p


def _kl(p, q):
    # Match P2's EPS smoothing and re-normalization, including that JS calls
    # this after applying its own smoothing.
    p = np.asarray(p, dtype=float) + 1e-12
    q = np.asarray(q, dtype=float) + 1e-12
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def _js(p, q):
    p, q = np.asarray(p, dtype=float) + 1e-12, np.asarray(q, dtype=float) + 1e-12
    p, q = p / p.sum(), q / q.sum()
    midpoint = (p + q) / 2.0
    return 0.5 * _kl(p, midpoint) + 0.5 * _kl(q, midpoint)


def features_from_updates(updates: Sequence, histograms) -> dict[str, np.ndarray]:
    """Compute P2's five signals from exchanged local update vectors and counts.

    Input order is peer order. Each update is the flattened effective adapter
    change during local training, not a factor vector or cumulative adapter.
    Different vector lengths are zero-padded as in P2, although aligned model
    layers normally produce equal lengths. Histograms must contain counts,
    not per-client normalized frequencies, to form the pooled reference.
    """
    vectors = [np.asarray(vector, dtype=float).reshape(-1) for vector in updates]
    if not vectors or any(vector.size == 0 or not np.isfinite(vector).all() for vector in vectors):
        raise ValueError("updates must contain nonempty finite vectors")
    counts = np.asarray(histograms, dtype=float)
    if (counts.ndim != 2 or counts.shape[0] != len(vectors) or counts.shape[1] == 0
            or not np.isfinite(counts).all() or np.any(counts < 0)
            or np.any(counts.sum(axis=1) <= 0)):
        raise ValueError("histograms must contain finite non-negative counts with positive mass per peer")
    width = max(vector.size for vector in vectors)
    matrix = np.vstack([np.pad(vector, (0, width - vector.size)) for vector in vectors])
    mean_update = matrix.mean(axis=0)
    mean_norm = np.linalg.norm(mean_update)
    pooled = counts.sum(axis=0)
    pooled /= pooled.sum()
    result = {name: [] for name in DOMAIN_POLICY_PROVENANCE["feature_names"]}
    for vector, histogram in zip(matrix, counts):
        probabilities = histogram / histogram.sum()
        nonzero = probabilities[probabilities > 0]
        entropy = float(-(nonzero * np.log(nonzero)).sum())
        positive = histogram[histogram > 0]
        imbalance = float(histogram.max() / positive.min()) * (1.0 + float(np.mean(histogram == 0)))
        denominator = np.linalg.norm(vector) * mean_norm
        cosine = 0.0 if denominator <= 1e-12 else float(1.0 - np.dot(vector, mean_update) / denominator)
        result["js_to_global"].append(_js(probabilities, pooled))
        result["update_l2_distance_to_mean"].append(float(np.linalg.norm(vector - mean_update)))
        result["update_cosine_distance_to_mean"].append(cosine)
        result["normalized_entropy"].append(entropy / np.log(counts.shape[1]) if counts.shape[1] > 1 else 0.0)
        result["class_imbalance_ratio"].append(imbalance)
    return {name: np.asarray(values, dtype=float) for name, values in result.items()}


def domain_features(states, histograms, alpha, *, reference_states=None) -> dict[str, np.ndarray]:
    """Extract scaled dense changes and compute P2's five domain signals.

    ``reference_states`` should be each peer's pre-local-training snapshot.
    If omitted, updates are measured from the frozen model's zero adapter.
    Sequences use positional order. A mapping of states fixes peer order by
    insertion order; histogram/reference mappings are aligned to those IDs.
    """
    if isinstance(states, Mapping):
        peer_ids = list(states)
        if isinstance(histograms, Mapping):
            if set(histograms) != set(peer_ids):
                raise ValueError("histograms and states must have the same peer IDs")
            histograms = [histograms[peer] for peer in peer_ids]
        if isinstance(reference_states, Mapping):
            if set(reference_states) != set(peer_ids):
                raise ValueError("reference_states and states must have the same peer IDs")
            reference_states = [reference_states[peer] for peer in peer_ids]
        states = [states[peer] for peer in peer_ids]
    else:
        states = list(states)
    if not states:
        raise ValueError("states must contain at least one peer")
    references = [None] * len(states) if reference_states is None else list(reference_states)
    if len(references) != len(states):
        raise ValueError("reference_states must contain one state per peer")
    layers = sorted(states[0])
    if not layers or any(sorted(state) != layers for state in states):
        raise ValueError("states must contain the same nonempty layer set")
    updates, shapes = [], None
    for state, reference in zip(states, references):
        effective = lora_to_delta(state, alpha)
        current_shapes = {name: tuple(effective[name].shape) for name in layers}
        if shapes is not None and current_shapes != shapes:
            raise ValueError("states must have matching effective layer dimensions")
        shapes = current_shapes
        if reference is not None:
            if sorted(reference) != layers:
                raise ValueError("reference_states must have the same layer set")
            before = lora_to_delta(reference, alpha)
            if any(tuple(before[name].shape) != shapes[name] for name in layers):
                raise ValueError("reference_states must have matching effective layer dimensions")
            effective = {name: effective[name] - before[name] for name in layers}
        updates.append(np.concatenate([effective[name].detach().cpu().double().numpy().reshape(-1)
                                       for name in layers]))
    return features_from_updates(updates, histograms)


__all__ = ["weighted_metropolis", "conservative_domain_factors", "DOMAIN_POLICY_PROVENANCE",
           "domain_features", "features_from_updates"]
