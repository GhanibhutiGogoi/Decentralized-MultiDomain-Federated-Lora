"""Unified evaluation metrics for Experiment 2 calibration outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from experiment2.lambda_calibration import (
    EPS,
    GROUP_COLS,
    TARGET,
    fit_form_a,
    fit_form_b,
    pearson,
    predict_delta_accuracy,
    predict_standardized_score,
    ridge_alpha_grid,
    spearman,
)


DEFAULT_PERMUTATION_SEED = 42
DEFAULT_PERMUTATIONS = 1000


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for deterministic statistical evaluation."""

    permutation_seed: int = DEFAULT_PERMUTATION_SEED
    permutations: int = DEFAULT_PERMUTATIONS
    group_cols: tuple[str, ...] = tuple(GROUP_COLS)


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    """Return standard regression metrics without changing model fitting."""
    actual = np.asarray(list(y_true), dtype=float)
    predicted = np.asarray(list(y_pred), dtype=float)
    if actual.size == 0:
        return {"rmse": 0.0, "mae": 0.0, "r_squared": 0.0, "pearson": 0.0}

    residual = actual - predicted
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > EPS else 0.0
    return {
        "rmse": rmse,
        "mae": mae,
        "r_squared": float(r_squared),
        "pearson": pearson(predicted, actual),
    }


def pairwise_ranking_accuracy(
    y_true: Iterable[float],
    y_score: Iterable[float],
) -> float:
    """Fraction of comparable item pairs ordered the same by score and target."""
    actual = np.asarray(list(y_true), dtype=float)
    score = np.asarray(list(y_score), dtype=float)
    comparable = 0
    correct = 0.0
    for i in range(len(actual)):
        for j in range(i + 1, len(actual)):
            actual_diff = actual[i] - actual[j]
            if abs(actual_diff) <= EPS:
                continue
            score_diff = score[i] - score[j]
            comparable += 1
            if abs(score_diff) <= EPS:
                correct += 0.5
            elif np.sign(actual_diff) == np.sign(score_diff):
                correct += 1.0
    return float(correct / comparable) if comparable else 0.0


def kendall_tau_b(
    y_true: Iterable[float],
    y_score: Iterable[float],
) -> float:
    """Compute Kendall tau-b with tie correction."""
    actual = np.asarray(list(y_true), dtype=float)
    score = np.asarray(list(y_score), dtype=float)
    concordant = 0
    discordant = 0
    ties_actual = 0
    ties_score = 0

    for i in range(len(actual)):
        for j in range(i + 1, len(actual)):
            actual_diff = actual[i] - actual[j]
            score_diff = score[i] - score[j]
            actual_tie = abs(actual_diff) <= EPS
            score_tie = abs(score_diff) <= EPS
            if actual_tie and score_tie:
                continue
            if actual_tie:
                ties_actual += 1
            elif score_tie:
                ties_score += 1
            elif np.sign(actual_diff) == np.sign(score_diff):
                concordant += 1
            else:
                discordant += 1

    denom = np.sqrt(
        (concordant + discordant + ties_actual)
        * (concordant + discordant + ties_score)
    )
    return float((concordant - discordant) / denom) if denom > EPS else 0.0


def permutation_rank_p_value(
    y_true: Iterable[float],
    y_score: Iterable[float],
    *,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_PERMUTATION_SEED,
) -> float:
    """One-sided permutation p-value for positive Spearman association."""
    actual = np.asarray(list(y_true), dtype=float)
    score = np.asarray(list(y_score), dtype=float)
    if actual.size < 2 or n_permutations <= 0:
        return 1.0

    observed = spearman(score, actual)
    if observed <= 0:
        return 1.0

    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(n_permutations):
        permuted = rng.permutation(actual)
        if spearman(score, permuted) >= observed - EPS:
            exceedances += 1
    return float((exceedances + 1) / (n_permutations + 1))


def ranking_metrics(
    y_true: Iterable[float],
    y_score: Iterable[float],
    *,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_PERMUTATION_SEED,
) -> dict[str, float]:
    """Return rank-based metrics and statistical significance."""
    actual = np.asarray(list(y_true), dtype=float)
    score = np.asarray(list(y_score), dtype=float)
    return {
        "spearman": spearman(score, actual),
        "pairwise_ranking_accuracy": pairwise_ranking_accuracy(actual, score),
        "kendall_tau": kendall_tau_b(actual, score),
        "permutation_p_value": permutation_rank_p_value(
            actual,
            score,
            n_permutations=n_permutations,
            seed=seed,
        ),
    }


def evaluate_predictions(
    df: pd.DataFrame,
    *,
    score_col: str,
    prediction_col: str = "predicted_delta_accuracy",
    target_col: str = TARGET,
    config: EvaluationConfig | None = None,
) -> dict[str, float]:
    """Compute the full unified metric set for a prediction table."""
    cfg = config or EvaluationConfig()
    metrics = regression_metrics(df[target_col], df[prediction_col])
    metrics.update(
        ranking_metrics(
            df[target_col],
            df[score_col],
            n_permutations=cfg.permutations,
            seed=cfg.permutation_seed,
        )
    )
    return metrics


def evaluation_table(
    lambda_values: pd.DataFrame,
    *,
    config: EvaluationConfig | None = None,
) -> pd.DataFrame:
    """Evaluate each form globally, per task, and per aggregation context."""
    cfg = config or EvaluationConfig()
    rows = []
    for form, form_df in lambda_values.groupby("form"):
        scopes = [("all", "ALL", form_df)]
        scopes.extend(("task", task, group) for task, group in form_df.groupby("task"))
        context_cols = list(cfg.group_cols)
        if set(context_cols).issubset(form_df.columns):
            for context_key, group in form_df.groupby(context_cols):
                context_name = "|".join(str(part) for part in context_key)
                scopes.append(("aggregation_context", context_name, group))

        for scope, scope_value, group in scopes:
            metrics = evaluate_predictions(
                group,
                score_col="raw_lambda_score",
                prediction_col="predicted_delta_accuracy",
                config=cfg,
            )
            rows.append(
                {
                    "form": form,
                    "scope": scope,
                    "scope_value": scope_value,
                    "n": len(group),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def leave_one_task_out_evaluation(
    df: pd.DataFrame,
    *,
    ridge_alphas: Iterable[float] | None = None,
    config: EvaluationConfig | None = None,
) -> tuple[pd.DataFrame, float]:
    """Evaluate held-out tasks while preserving RMSE alpha selection."""
    cfg = config or EvaluationConfig()
    rows = []
    alphas = ridge_alpha_grid(custom_alphas=ridge_alphas)

    for held_out in sorted(df["task"].unique()):
        train = df[df["task"] != held_out].reset_index(drop=True)
        test = df[df["task"] == held_out].reset_index(drop=True)

        fit_a = fit_form_a(train)
        pred_a = predict_delta_accuracy(test, fit_a)
        score_a = predict_standardized_score(test, fit_a)
        metrics_a = regression_metrics(test[TARGET], pred_a)
        metrics_a.update(
            ranking_metrics(
                test[TARGET],
                score_a,
                n_permutations=cfg.permutations,
                seed=cfg.permutation_seed,
            )
        )
        rows.append(
            {
                "form": "form_a",
                "held_out_task": held_out,
                "ridge_alpha": "",
                "n_train": len(train),
                "n_test": len(test),
                **metrics_a,
            }
        )

        for alpha in alphas:
            fit_b = fit_form_b(train, alpha)
            pred_b = predict_delta_accuracy(test, fit_b)
            score_b = predict_standardized_score(test, fit_b)
            metrics_b = regression_metrics(test[TARGET], pred_b)
            metrics_b.update(
                ranking_metrics(
                    test[TARGET],
                    score_b,
                    n_permutations=cfg.permutations,
                    seed=cfg.permutation_seed,
                )
            )
            rows.append(
                {
                    "form": "form_b",
                    "held_out_task": held_out,
                    "ridge_alpha": alpha,
                    "n_train": len(train),
                    "n_test": len(test),
                    **metrics_b,
                }
            )

    cv = pd.DataFrame(rows)
    ridge_rows = cv[cv["form"] == "form_b"].copy()
    mean_rmse = ridge_rows.groupby("ridge_alpha")["rmse"].mean()
    selected_alpha = float(mean_rmse.idxmin())
    return cv, selected_alpha


def alpha_evaluation_table(
    cv: pd.DataFrame,
    *,
    config: EvaluationConfig | None = None,
) -> pd.DataFrame:
    """Summarize held-out alpha candidates without changing selection logic."""
    del config
    numeric = [
        "rmse",
        "mae",
        "r_squared",
        "pearson",
        "spearman",
        "pairwise_ranking_accuracy",
        "kendall_tau",
        "permutation_p_value",
    ]
    available = [col for col in numeric if col in cv.columns]
    group_cols = ["form", "ridge_alpha"]
    rows = (
        cv.groupby(group_cols, dropna=False)[available]
        .mean()
        .reset_index()
    )
    rows["_ridge_alpha_sort"] = pd.to_numeric(
        rows["ridge_alpha"], errors="coerce"
    ).fillna(-1.0)
    rows = rows.sort_values(["form", "_ridge_alpha_sort"]).drop(
        columns=["_ridge_alpha_sort"]
    )
    return rows
