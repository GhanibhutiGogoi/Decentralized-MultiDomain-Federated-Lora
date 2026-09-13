"""Experiment 3 metrics with strict table validation."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


class Experiment3MetricError(ValueError):
    """Raised when metric inputs are missing or malformed."""


REQUIRED_OBSERVATION_COLUMNS = (
    "task",
    "seed",
    "arm",
    "round",
    "client_id",
    "domain",
    "accuracy",
)


def validate_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Validate metric observations without dropping, coercing, or imputing rows."""
    missing = [column for column in REQUIRED_OBSERVATION_COLUMNS if column not in df.columns]
    if missing:
        raise Experiment3MetricError(f"Missing required metric columns: {missing}.")
    if df.empty:
        raise Experiment3MetricError("Metric observations are empty.")
    duplicate_keys = ["task", "seed", "arm", "round", "client_id", "domain"]
    if df.duplicated(duplicate_keys).any():
        raise Experiment3MetricError("Metric observations contain duplicate identifiers.")
    out = df.copy()
    for column in ("round", "accuracy"):
        invalid = []
        parsed_values = []
        for index, value in out[column].items():
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                invalid.append(index)
                continue
            if not math.isfinite(parsed):
                invalid.append(index)
            parsed_values.append(parsed)
        if invalid:
            raise Experiment3MetricError(
                f"Column {column!r} contains missing, NaN, infinite, or malformed values."
            )
        out[column] = parsed_values
    return out


def global_accuracy_per_round(df: pd.DataFrame) -> pd.DataFrame:
    """Mean global accuracy for each task, seed, arm, and round."""
    out = validate_observations(df)
    return (
        out.groupby(["task", "seed", "arm", "round"], as_index=False)["accuracy"]
        .mean()
        .rename(columns={"accuracy": "global_accuracy"})
    )


def final_round_global_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Global accuracy from the maximum observed round in each task, seed, and arm."""
    curve = global_accuracy_per_round(df)
    max_round = curve.groupby(["task", "seed", "arm"])["round"].transform("max")
    return curve[curve["round"] == max_round].reset_index(drop=True)


def convergence_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Per-round global accuracy curve preserving task, seed, arm, and round."""
    return global_accuracy_per_round(df).sort_values(
        ["task", "seed", "arm", "round"],
        kind="mergesort",
    )


def convergence_speed(df: pd.DataFrame, target_accuracy: float) -> pd.DataFrame:
    """First round whose global accuracy is at least ``target_accuracy``.

    If a group never reaches the target, ``round_to_target`` is ``None``.
    """
    target = float(target_accuracy)
    if not math.isfinite(target):
        raise Experiment3MetricError("target_accuracy must be finite.")
    rows = []
    for key, group in convergence_curve(df).groupby(["task", "seed", "arm"]):
        reached = group[group["global_accuracy"] >= target]
        rows.append(
            {
                "task": key[0],
                "seed": key[1],
                "arm": key[2],
                "target_accuracy": target,
                "round_to_target": None
                if reached.empty
                else int(reached.sort_values("round").iloc[0]["round"]),
            }
        )
    return pd.DataFrame(rows)


def per_domain_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Mean accuracy per task, seed, arm, round, and domain."""
    out = validate_observations(df)
    return (
        out.groupby(["task", "seed", "arm", "round", "domain"], as_index=False)[
            "accuracy"
        ]
        .mean()
        .rename(columns={"accuracy": "domain_accuracy"})
    )


def worst_domain_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Minimum per-domain accuracy per task, seed, arm, and round."""
    per_domain = per_domain_accuracy(df)
    return (
        per_domain.groupby(["task", "seed", "arm", "round"], as_index=False)[
            "domain_accuracy"
        ]
        .min()
        .rename(columns={"domain_accuracy": "worst_domain_accuracy"})
    )


def fairness_spread_variance(df: pd.DataFrame) -> pd.DataFrame:
    """Population variance of per-domain accuracy within each task, seed, arm, round."""
    per_domain = per_domain_accuracy(df)
    return (
        per_domain.groupby(["task", "seed", "arm", "round"], as_index=False)[
            "domain_accuracy"
        ]
        .var(ddof=0)
        .rename(columns={"domain_accuracy": "domain_accuracy_variance"})
    )


def realized_client_weights(
    records: Sequence[dict],
    weights: Sequence[float],
) -> pd.DataFrame:
    """Validated realized client aggregation weights for one round and arm."""
    if len(records) != len(weights):
        raise Experiment3MetricError("records and weights have different lengths.")
    rows = []
    seen = set()
    for record, weight in zip(records, weights):
        key = (
            record.get("task"),
            record.get("seed"),
            record.get("arm"),
            record.get("round"),
            record.get("client_id"),
        )
        if any(value is None or value == "" for value in key):
            raise Experiment3MetricError("Weight record is missing an identifier.")
        if key in seen:
            raise Experiment3MetricError("Weight records contain duplicate identifiers.")
        seen.add(key)
        value = float(weight)
        if not math.isfinite(value) or value < 0.0:
            raise Experiment3MetricError("Weights must be finite and non-negative.")
        rows.append({**record, "realized_aggregation_weight": value})
    total = sum(row["realized_aggregation_weight"] for row in rows)
    if total <= 0.0:
        raise Experiment3MetricError("At least one positive realized weight is required.")
    return pd.DataFrame(rows)


def _rankdata(values: Iterable[float]) -> np.ndarray:
    series = pd.Series(list(values), dtype=float)
    return series.rank(method="average").to_numpy(dtype=float)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def within_round_rank_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman rank agreement between lambda and intended divergence ordering."""
    required = [
        "task",
        "seed",
        "arm",
        "round",
        "client_id",
        "lambda_weight",
        "intended_domain_divergence_order",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise Experiment3MetricError(f"Missing rank-agreement columns: {missing}.")
    rows = []
    for key, group in df.groupby(["task", "seed", "arm", "round"], sort=False):
        if len(group) < 2:
            rho = 0.0
        else:
            lam = group["lambda_weight"].to_numpy(dtype=float)
            order = group["intended_domain_divergence_order"].to_numpy(dtype=float)
            if not np.all(np.isfinite(lam)) or not np.all(np.isfinite(order)):
                raise Experiment3MetricError("Rank agreement inputs must be finite.")
            rho = _pearson(_rankdata(lam), _rankdata(order))
        rows.append(
            {
                "task": key[0],
                "seed": key[1],
                "arm": key[2],
                "round": key[3],
                "lambda_divergence_spearman": rho,
            }
        )
    return pd.DataFrame(rows)

