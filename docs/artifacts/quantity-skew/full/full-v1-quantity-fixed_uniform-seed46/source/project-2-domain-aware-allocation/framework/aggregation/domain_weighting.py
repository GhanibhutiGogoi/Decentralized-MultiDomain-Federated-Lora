"""Conservative, evidence-weighted domain allocation.

The original Project 2 lambda formulation can move a client's aggregation
weight by as much as 3x (the ``[0.5, 1.5]`` bounds) even when the domain signal
is noisy.  This module provides a trust-region wrapper for deployment: domain
signals are standardised, converted to a positive factor, and then shrunk
towards one.  The resulting factor is normalised under the *base* FedAvg
weights, so enabling domain awareness cannot change the update scale and can
only make a bounded redistribution between clients.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


EPS = 1e-12


def _as_array(values, name: str, n: int) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1 or arr.size != n:
        raise ValueError(f"{name} must contain exactly {n} values.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _standardise(values: np.ndarray) -> np.ndarray:
    center = float(values.mean())
    scale = float(values.std(ddof=0))
    if scale <= EPS:
        return np.zeros_like(values)
    # Robustly cap extreme observations so a single client cannot dominate.
    z = (values - center) / scale
    return np.clip(z, -3.0, 3.0)


def conservative_domain_factors(
    samples: Sequence[float],
    quality_scores: Sequence[float],
    domain_features: Mapping[str, Sequence[float]] | None = None,
    *,
    blend_strength: float = 0.10,
    max_deviation: float = 0.15,
    temperature: float = 1.0,
) -> np.ndarray:
    """Return bounded, base-weight-normalised domain factors.

    ``domain_features`` maps signal names (for example ``js_to_global`` or
    ``update_l2``) to one value per client.  Signals are z-scored and averaged
    before an exponential map.  The map is shrunk by ``blend_strength`` and a
    fixed trust region ``max_deviation``.  Finally, factors are renormalised so
    ``sum_i samples_i*q_i*factor_i == sum_i samples_i*q_i``.

    With no features, constant features, or zero blend strength the function
    returns all ones, exactly preserving the existing quality-weighted FedAvg
    path.
    """
    n = len(samples)
    if n == 0:
        raise ValueError("at least one client is required")
    samples_arr = _as_array(samples, "samples", n)
    quality_arr = _as_array(quality_scores, "quality_scores", n)
    if np.any(samples_arr < 0) or np.any(quality_arr < 0):
        raise ValueError("samples and quality_scores must be non-negative")
    base = samples_arr * quality_arr
    if not np.isfinite(base).all() or float(base.sum()) <= EPS:
        raise ValueError("samples*quality_scores must have a positive total")
    if not (0.0 <= blend_strength <= 1.0):
        raise ValueError("blend_strength must lie in [0, 1]")
    if not (0.0 < max_deviation < 1.0):
        raise ValueError("max_deviation must lie in (0, 1)")
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be positive and finite")

    if not domain_features or blend_strength == 0.0:
        return np.ones(n, dtype=float)
    columns = []
    for name, values in domain_features.items():
        columns.append(_standardise(_as_array(values, f"domain feature {name}", n)))
    if not columns:
        return np.ones(n, dtype=float)
    score = np.mean(np.vstack(columns), axis=0)
    if float(np.max(np.abs(score))) <= EPS:
        return np.ones(n, dtype=float)

    raw = np.exp(np.clip(score / float(temperature), -4.0, 4.0))
    raw /= max(float(np.average(raw, weights=base)), EPS)
    # Trust-region interpolation.  This is deliberately conservative: the
    # default can move factors by at most 3.75 percentage points.
    factors = 1.0 + float(blend_strength) * (raw - 1.0)
    lower, upper = 1.0 - max_deviation, 1.0 + max_deviation
    # Find a scalar multiplier whose clipped weighted mean is exactly one.
    # This is a monotone one-dimensional problem and avoids the common
    # clip-then-renormalise bug where the second operation violates bounds.
    def mean_at(scale: float) -> float:
        return float(np.average(np.clip(factors * scale, lower, upper), weights=base))

    lo, hi = 0.0, 1.0
    while mean_at(hi) < 1.0:
        hi *= 2.0
        if hi > 1e12:
            raise RuntimeError("unable to normalise domain factors")
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if mean_at(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    return np.clip(factors * ((lo + hi) / 2.0), lower, upper)


__all__ = ["conservative_domain_factors"]
