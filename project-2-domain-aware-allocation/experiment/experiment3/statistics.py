"""Paired statistical utilities for Experiment 3 engineering validation."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


class Experiment3StatisticsError(ValueError):
    """Raised when a statistical procedure is not properly paired."""


@dataclass(frozen=True)
class PermutationResult:
    """Paired sign-flip permutation result."""

    observed_mean_difference: float
    p_value: float
    n_pairs: int
    method: str
    alternative: str
    permutations: int


def paired_arm_differences(
    df: pd.DataFrame,
    *,
    baseline_arm: str,
    comparison_arm: str,
    value_col: str,
    unit_cols: Sequence[str] = ("task", "seed", "partition_hash"),
) -> pd.DataFrame:
    """Return comparison-minus-baseline values paired by experimental unit."""
    required = [*unit_cols, "arm", value_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise Experiment3StatisticsError(f"Missing paired-difference columns: {missing}.")
    subset = df[df["arm"].isin([baseline_arm, comparison_arm])].copy()
    if subset.empty:
        raise Experiment3StatisticsError("No matching arm observations.")
    if subset.duplicated([*unit_cols, "arm"]).any():
        raise Experiment3StatisticsError("Duplicate paired observations detected.")
    wide = subset.pivot(index=list(unit_cols), columns="arm", values=value_col)
    if baseline_arm not in wide.columns or comparison_arm not in wide.columns:
        raise Experiment3StatisticsError("Both baseline and comparison arms are required.")
    if wide[[baseline_arm, comparison_arm]].isna().any().any():
        raise Experiment3StatisticsError("Pairing is incomplete.")
    result = wide.reset_index()
    result["difference"] = result[comparison_arm] - result[baseline_arm]
    if not np.all(np.isfinite(result["difference"].to_numpy(dtype=float))):
        raise Experiment3StatisticsError("Paired differences must be finite.")
    return result[list(unit_cols) + [baseline_arm, comparison_arm, "difference"]]


def _tail_count(values: np.ndarray, observed: float, alternative: str) -> int:
    if alternative == "two-sided":
        return int(np.sum(np.abs(values) >= abs(observed) - 1e-15))
    if alternative == "greater":
        return int(np.sum(values >= observed - 1e-15))
    if alternative == "less":
        return int(np.sum(values <= observed + 1e-15))
    raise Experiment3StatisticsError("alternative must be two-sided, greater, or less.")


def paired_permutation_test(
    differences: Sequence[float],
    *,
    seed: int,
    alternative: str = "two-sided",
    max_exact_pairs: int = 16,
    monte_carlo_permutations: int = 10000,
) -> PermutationResult:
    """Paired sign-flip permutation preserving experimental-unit pairing.

    Exact enumeration is used for small ``n``. Monte Carlo uses the plus-one
    convention: ``(extreme + 1) / (permutations + 1)``.
    """
    diffs = np.asarray(list(differences), dtype=float)
    if diffs.size == 0:
        raise Experiment3StatisticsError("At least one paired difference is required.")
    if not np.all(np.isfinite(diffs)):
        raise Experiment3StatisticsError("Paired differences must be finite.")
    observed = float(diffs.mean())
    n = len(diffs)

    if n <= max_exact_pairs:
        means = []
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            means.append(float(np.mean(diffs * np.asarray(signs))))
        null = np.asarray(means, dtype=float)
        p_value = _tail_count(null, observed, alternative) / len(null)
        method = "exact"
        permutations = len(null)
    else:
        rng = np.random.default_rng(int(seed))
        signs = rng.choice([-1.0, 1.0], size=(int(monte_carlo_permutations), n))
        null = np.mean(signs * diffs, axis=1)
        extreme = _tail_count(null, observed, alternative)
        p_value = (extreme + 1.0) / (int(monte_carlo_permutations) + 1.0)
        method = "monte_carlo_plus_one"
        permutations = int(monte_carlo_permutations)

    return PermutationResult(
        observed_mean_difference=observed,
        p_value=float(p_value),
        n_pairs=n,
        method=method,
        alternative=alternative,
        permutations=permutations,
    )


def confidence_interval(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
) -> tuple[float | None, float | None]:
    """Student-t interval for a paired mean; undefined for a single pair.

    The interval assumes independent paired differences. With one pair their
    sampling variance cannot be estimated, so export missing bounds instead
    of a zero-width interval that would imply certainty.
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise Experiment3StatisticsError("CI values must be finite and non-empty.")
    if not 0.0 < confidence < 1.0:
        raise Experiment3StatisticsError("confidence must be between 0 and 1.")
    if arr.size == 1:
        return None, None
    quantile = float(student_t.ppf((1 + confidence) / 2, df=len(arr) - 1))
    mean = float(arr.mean())
    margin = quantile * float(arr.std(ddof=1)) / math.sqrt(len(arr))
    return mean - margin, mean + margin


def compare_to_mde(effect: float, *, mde: float | None) -> dict[str, float | bool]:
    """Compare an observed effect to a required configured MDE.

    No default MDE is invented; callers must provide one when this comparison is
    scientifically authorized.
    """
    if mde is None:
        raise Experiment3StatisticsError("MDE must be configured before comparison.")
    effect_value = float(effect)
    mde_value = float(mde)
    if not math.isfinite(effect_value) or not math.isfinite(mde_value) or mde_value <= 0:
        raise Experiment3StatisticsError("effect and positive mde must be finite.")
    return {
        "effect": effect_value,
        "mde": mde_value,
        "meets_mde": abs(effect_value) >= mde_value,
    }


def task_level_summary(
    paired: pd.DataFrame,
    *,
    task_col: str = "task",
    difference_col: str = "difference",
) -> pd.DataFrame:
    """Summarize paired differences by task without treating rounds as pairs."""
    if task_col not in paired.columns or difference_col not in paired.columns:
        raise Experiment3StatisticsError("Task summary columns are missing.")
    return (
        paired.groupby(task_col, as_index=False)[difference_col]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "n_pairs", "mean": "mean_difference"})
    )


def pooled_summary(paired: pd.DataFrame, *, difference_col: str = "difference") -> dict:
    """Summarize all valid paired experimental units as one paired sample."""
    if difference_col not in paired.columns:
        raise Experiment3StatisticsError("Pooled summary difference column is missing.")
    diffs = paired[difference_col].to_numpy(dtype=float)
    if len(diffs) == 0 or not np.all(np.isfinite(diffs)):
        raise Experiment3StatisticsError("Pooled paired differences are invalid.")
    low, high = confidence_interval(diffs)
    return {
        "n_pairs": int(len(diffs)),
        "mean_difference": float(np.mean(diffs)),
        "ci_low": low,
        "ci_high": high,
    }
