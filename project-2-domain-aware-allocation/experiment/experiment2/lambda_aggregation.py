"""Optional lambda extension for quality-weighted aggregation.

This module does not replace Project 1 aggregation. When lambda_weights is
None, the wrapped aggregation receives the original quality scores unchanged.
When lambda_weights is supplied, the base aggregator sees q_i * lambda_i, so
its existing samples * q_i formula becomes samples * q_i * lambda_i.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from framework.aggregation.domain_weighting import conservative_domain_factors


def effective_quality_scores(quality_scores, lambda_weights=None):
    """Return q or q * lambda while preserving q exactly when disabled."""
    if lambda_weights is None:
        return list(quality_scores)
    if len(quality_scores) != len(lambda_weights):
        raise ValueError("quality_scores and lambda_weights must have equal length.")
    return [
        float(quality) * float(lambda_weight)
        for quality, lambda_weight in zip(quality_scores, lambda_weights)
    ]


def normalized_aggregation_weights(samples, quality_scores, lambda_weights=None):
    """Compute normalized samples * q * lambda aggregation weights."""
    qualities = effective_quality_scores(quality_scores, lambda_weights)
    raw = [float(sample_count) * quality for sample_count, quality in zip(samples, qualities)]
    total = sum(raw)
    if total <= 0:
        return [1.0 / len(raw)] * len(raw)
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
