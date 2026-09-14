"""Optional lambda extension for quality-weighted aggregation.

This module does not replace Project 1 aggregation. When lambda_weights is
None, the wrapped aggregation receives the original quality scores unchanged.
When lambda_weights is supplied, the base aggregator sees q_i * lambda_i, so
its existing samples * q_i formula becomes samples * q_i * lambda_i.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from framework.aggregation.domain_weighting import conservative_domain_factors


def _validated_nonnegative_sequence(values, name: str) -> list[float]:
    if values is None:
        raise ValueError(f"{name} is required.")
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a numeric sequence.")
    parsed = []
    for idx, value in enumerate(values):
        if isinstance(value, bool):
            raise ValueError(f"{name}[{idx}] must be numeric.")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{idx}] must be numeric.") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name}[{idx}] must be finite.")
        if number < 0.0:
            raise ValueError(f"{name}[{idx}] must be non-negative.")
        parsed.append(number)
    if not parsed:
        raise ValueError(f"{name} must contain at least one value.")
    return parsed


def effective_quality_scores(quality_scores, lambda_weights=None):
    """Return q or q * lambda while preserving q exactly when disabled."""
    qualities = _validated_nonnegative_sequence(quality_scores, "quality_scores")
    if lambda_weights is None:
        return qualities
    lambdas = _validated_nonnegative_sequence(lambda_weights, "lambda_weights")
    if len(qualities) != len(lambdas):
        raise ValueError("quality_scores and lambda_weights must have equal length.")
    return [
        quality * lambda_weight
        for quality, lambda_weight in zip(qualities, lambdas)
    ]


def normalized_aggregation_weights(samples, quality_scores, lambda_weights=None, *, client_ids=None):
    """Compute normalized samples * q * lambda aggregation weights."""
    sample_values = _validated_nonnegative_sequence(samples, "samples")
    qualities = effective_quality_scores(quality_scores, lambda_weights)
    if len(sample_values) != len(qualities):
        raise ValueError("samples and quality_scores must have equal length.")
    if client_ids is not None:
        client_ids = [str(value) for value in client_ids]
        if len(client_ids) != len(sample_values):
            raise ValueError("client_ids must match samples length.")
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("client_ids must be unique.")
    raw = [sample_count * quality for sample_count, quality in zip(sample_values, qualities)]
    total = sum(raw)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("samples*quality*lambda must have a finite, positive total.")
    return [value / total for value in raw]


def normalized_conservative_domain_weights(
    samples: Sequence[float],
    quality_scores: Sequence[float],
    domain_features: Mapping[str, Sequence[float]] | None = None,
    *,
    blend_strength: float = 0.10,
    max_deviation: float = 0.15,
    temperature: float = 1.0,
):
    """Quality weights with a bounded, conservative domain redistribution.

    The return value is directly usable by the aggregation routines.  Domain
    features are optional; omitting them gives the original ``samples*q``
    weights exactly.
    """
    factors = conservative_domain_factors(
        samples,
        quality_scores,
        domain_features,
        blend_strength=blend_strength,
        max_deviation=max_deviation,
        temperature=temperature,
    )
    raw = [
        float(sample) * float(quality) * float(factor)
        for sample, quality, factor in zip(samples, quality_scores, factors)
    ]
    total = sum(raw)
    if total <= 0:
        raise ValueError("samples*quality*domain factors must have positive total")
    return [value / total for value in raw]


def fedavg_quality_lambda_weighted(
    weights,
    samples,
    quality_scores,
    target_rank,
    ref_sd,
    device,
    lambda_weights=None,
    base_aggregate_fn=None,
):
    """Call an existing FedAvg implementation with optional q * lambda scores."""
    if base_aggregate_fn is None:
        from framework.aggregation import fedavg_quality_weighted as base_aggregate_fn

    adjusted_quality = effective_quality_scores(quality_scores, lambda_weights)
    return base_aggregate_fn(
        weights,
        samples,
        adjusted_quality,
        target_rank,
        ref_sd,
        device,
    )
