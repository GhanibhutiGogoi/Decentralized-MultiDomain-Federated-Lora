"""Benchmark domain-weight policies on Experiment 1 contribution records.

This is a model-free check of the allocation rule.  It uses the measured
leave-one-client-out contribution as an offline target and compares the
existing sample*quality rule with the conservative domain-aware extension.
It deliberately reports ranking and weighted-contribution metrics rather than
claiming end-to-end training accuracy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from framework.aggregation.domain_weighting import conservative_domain_factors
except ModuleNotFoundError as exc:  # lightweight offline benchmark without torch
    if exc.name not in {"torch", "framework"}:
        raise
    import importlib.util

    _path = Path(__file__).resolve().parents[2] / "framework" / "aggregation" / "domain_weighting.py"
    if not _path.exists():
        _path = Path(__file__).resolve().with_name("domain_weighting.py")
    _spec = importlib.util.spec_from_file_location("_domain_weighting", _path)
    _module = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_module)
    conservative_domain_factors = _module.conservative_domain_factors


def _spearman(x, y):
    return float(pd.Series(x).rank().corr(pd.Series(y).rank())) if len(x) > 1 else 0.0


def _pairwise_accuracy(actual, score):
    good = total = 0
    for i in range(len(actual)):
        for j in range(i + 1, len(actual)):
            if actual[i] == actual[j]:
                continue
            total += 1
            good += int(np.sign(actual[i] - actual[j]) == np.sign(score[i] - score[j]))
    return float(good / total) if total else 0.0


def benchmark(measurements: pd.DataFrame, blend_strength: float = 0.10):
    rows = []
    feature_cols = [
        "predicted_delta_accuracy",
        "js_to_global",
        "update_l2_distance_to_mean",
        "update_cosine_distance_to_mean",
        "normalized_entropy",
        "class_imbalance_ratio",
    ]
    for (task, round_id), group in measurements.groupby(["task", "round"]):
        g = group.reset_index(drop=True)
        samples = g["train_samples_seen"].to_numpy(float)
        quality = g["quality_score"].to_numpy(float)
        base_raw = samples * quality
        base = base_raw / base_raw.sum()
        if "predicted_delta_accuracy" in g and g["predicted_delta_accuracy"].notna().all():
            features = {"predicted_delta_accuracy": g["predicted_delta_accuracy"].to_numpy(float)}
        else:
            features = {name: g[name].to_numpy(float) for name in feature_cols[1:] if name in g}
        factors = conservative_domain_factors(
            samples,
            quality,
            features,
            blend_strength=blend_strength,
        )
        domain_raw = base_raw * factors
        domain = domain_raw / domain_raw.sum()
        target = g["delta_accuracy"].to_numpy(float)
        rows.extend(
            [
                {
                    "task": task,
                    "round": int(round_id),
                    "policy": "quality",
                    "spearman_delta": _spearman(base, target),
                    "pairwise_accuracy": _pairwise_accuracy(target, base),
                    "weighted_delta": float(np.dot(base, target)),
                    "max_weight": float(base.max()),
                },
                {
                    "task": task,
                    "round": int(round_id),
                    "policy": "conservative_domain",
                    "spearman_delta": _spearman(domain, target),
                    "pairwise_accuracy": _pairwise_accuracy(target, domain),
                    "weighted_delta": float(np.dot(domain, target)),
                    "max_weight": float(domain.max()),
                },
            ]
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Optional lambda_values.csv containing calibrated predictions.",
    )
    parser.add_argument("--blend-strength", type=float, default=0.10)
    args = parser.parse_args()
    measurements = pd.read_csv(args.measurements)
    if args.predictions is not None:
        predictions = pd.read_csv(args.predictions)
        keep = ["task", "round", "client_id", "predicted_delta_accuracy"]
        if not set(keep).issubset(predictions.columns):
            raise ValueError("predictions file lacks the required columns")
        # Use Form A as the primary calibrated predictor; it is the selected
        # low-variance candidate in Experiment 2.
        if "form" in predictions.columns:
            predictions = predictions[predictions["form"] == "form_a"]
        measurements = measurements.merge(
            predictions[keep], on=["task", "round", "client_id"], how="left"
        )
    required = {"task", "round", "train_samples_seen", "quality_score", "delta_accuracy"}
    missing = required - set(measurements.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    result = benchmark(measurements, args.blend_strength)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    summary = result.groupby("policy")[["spearman_delta", "pairwise_accuracy", "weighted_delta", "max_weight"]].mean()
    print(summary.to_string())


if __name__ == "__main__":
    main()
