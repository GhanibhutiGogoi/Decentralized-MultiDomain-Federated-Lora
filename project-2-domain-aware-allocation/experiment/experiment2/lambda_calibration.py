"""Calibration utilities for Experiment 2 domain-aware lambda."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-12
TARGET = "delta_accuracy"
QUALITY = "quality_score"
GROUP_COLS = ["task", "round"]

FORM_A_FEATURES = ["log_update_l2", "js_to_global"]
FORM_B_FEATURES = [
    "log_update_l2",
    "js_to_global",
    "update_cosine_distance_to_mean",
    "normalized_entropy",
    "log_class_imbalance_ratio",
]
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


@dataclass(frozen=True)
class LinearFit:
    form: str
    features: list[str]
    coefficients: np.ndarray
    intercept: float
    feature_means: dict[str, float]
    feature_stds: dict[str, float]
    target_mean: float
    target_std: float
    ridge_alpha: float | None = None


def prepare_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Create stable transformed features used by both lambda forms."""
    out = df.copy()
    out["log_update_l2"] = np.log1p(out["update_l2_distance_to_mean"].clip(lower=0.0))
    out["log_class_imbalance_ratio"] = np.log1p(
        out["class_imbalance_ratio"].clip(lower=0.0)
    )
    needed = sorted(set([TARGET, QUALITY, *GROUP_COLS, *FORM_B_FEATURES]))
    return out.dropna(subset=[col for col in needed if col in out]).reset_index(drop=True)


def pearson(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    if len(x_arr) < 2 or x_arr.std() <= EPS or y_arr.std() <= EPS:
        return 0.0
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    x_ser = pd.Series(list(x), dtype=float)
    y_ser = pd.Series(list(y), dtype=float)
    if len(x_ser) < 2 or x_ser.nunique() < 2 or y_ser.nunique() < 2:
        return 0.0
    return pearson(x_ser.rank(method="average"), y_ser.rank(method="average"))


def _standardization_stats(df: pd.DataFrame, features: list[str]):
    means = {feature: float(df[feature].mean()) for feature in features}
    stds = {}
    for feature in features:
        std = float(df[feature].std(ddof=0))
        stds[feature] = std if std > EPS else 1.0
    return means, stds


def _standardize_features(
    df: pd.DataFrame,
    features: list[str],
    means: dict[str, float],
    stds: dict[str, float],
) -> np.ndarray:
    return np.column_stack(
        [
            (df[feature].to_numpy(dtype=float) - means[feature]) / stds[feature]
            for feature in features
        ]
    )


def _standardize_target(df: pd.DataFrame):
    y = df[TARGET].to_numpy(dtype=float)
    mean = float(y.mean())
    std = float(y.std(ddof=0))
    if std <= EPS:
        std = 1.0
    return (y - mean) / std, mean, std


def fit_form_a(df: pd.DataFrame) -> LinearFit:
    """Fit the interpretable two-factor formula with OLS."""
    features = FORM_A_FEATURES
    means, stds = _standardization_stats(df, features)
    x = _standardize_features(df, features, means, stds)
    y, y_mean, y_std = _standardize_target(df)
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return LinearFit(
        form="form_a",
        features=features,
        coefficients=beta[1:],
        intercept=float(beta[0]),
        feature_means=means,
        feature_stds=stds,
        target_mean=y_mean,
        target_std=y_std,
    )


def fit_form_b(df: pd.DataFrame, ridge_alpha: float) -> LinearFit:
    """Fit the data-driven interpretable ridge formulation."""
    features = FORM_B_FEATURES
    means, stds = _standardization_stats(df, features)
    x = _standardize_features(df, features, means, stds)
    y, y_mean, y_std = _standardize_target(df)
    xtx = x.T @ x
    xty = x.T @ y
    coef = np.linalg.solve(xtx + ridge_alpha * np.eye(x.shape[1]), xty)
    return LinearFit(
        form="form_b",
        features=features,
        coefficients=coef,
        intercept=0.0,
        feature_means=means,
        feature_stds=stds,
        target_mean=y_mean,
        target_std=y_std,
        ridge_alpha=ridge_alpha,
    )


def predict_standardized_score(df: pd.DataFrame, fit: LinearFit) -> np.ndarray:
    x = _standardize_features(df, fit.features, fit.feature_means, fit.feature_stds)
    return fit.intercept + x @ fit.coefficients


def predict_delta_accuracy(df: pd.DataFrame, fit: LinearFit) -> np.ndarray:
    return fit.target_mean + fit.target_std * predict_standardized_score(df, fit)


def _lambda_from_score(df: pd.DataFrame, score: np.ndarray, scale: float):
    work = df[GROUP_COLS].copy()
    work["score"] = score
    centered = work["score"] - work.groupby(GROUP_COLS)["score"].transform("mean")
    raw = np.exp(scale * centered.clip(lower=-20.0, upper=20.0))
    work["lambda_raw"] = raw
    raw_mean = work.groupby(GROUP_COLS)["lambda_raw"].transform("mean")
    lam = work["lambda_raw"] / raw_mean
    lam = lam.clip(lower=0.5, upper=1.5)
    work["lambda_clipped"] = lam
    clipped_mean = work.groupby(GROUP_COLS)["lambda_clipped"].transform("mean")
    return (work["lambda_clipped"] / clipped_mean).to_numpy(dtype=float)


def calibrate_lambda_scale(df: pd.DataFrame, scores_by_form: dict[str, np.ndarray]) -> float:
    """Choose one shared scale so lambda has lower CV than q."""
    q_cv = float(df[QUALITY].std(ddof=0) / max(abs(df[QUALITY].mean()), EPS))
    target_cv = min(0.5 * q_cv, 0.20)
    if target_cv <= EPS:
        return 0.0

    all_scores = np.concatenate(list(scores_by_form.values()))
    repeated = pd.concat([df[GROUP_COLS]] * len(scores_by_form), ignore_index=True)

    lo, hi = 0.0, 5.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        lam = _lambda_from_score(repeated, all_scores, mid)
        cv = float(lam.std(ddof=0) / max(abs(lam.mean()), EPS))
        if cv > target_cv:
            hi = mid
        else:
            lo = mid
    return lo


def attach_lambda_values(
    df: pd.DataFrame,
    fits: list[LinearFit],
    lambda_scale: float,
) -> pd.DataFrame:
    rows = []
    base_cols = [
        "task",
        "round",
        "client_id",
        "quality_score",
        "delta_accuracy",
        "js_to_global",
        "update_l2_distance_to_mean",
        "update_cosine_distance_to_mean",
        "normalized_entropy",
        "class_imbalance_ratio",
    ]
    for fit in fits:
        score = predict_standardized_score(df, fit)
        lam = _lambda_from_score(df, score, lambda_scale)
        pred = predict_delta_accuracy(df, fit)
        part = df[base_cols].copy()
        part["form"] = fit.form
        part["raw_lambda_score"] = score
        part["predicted_delta_accuracy"] = pred
        part["lambda_weight"] = lam
        part["effective_quality_score"] = part["quality_score"] * part["lambda_weight"]
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def coefficient_table(fits: list[LinearFit]) -> pd.DataFrame:
    rows = []
    for fit in fits:
        rows.append(
            {
                "form": fit.form,
                "term": "intercept",
                "coefficient": fit.intercept,
                "abs_coefficient": abs(fit.intercept),
                "ridge_alpha": fit.ridge_alpha,
                "feature_mean": "",
                "feature_std": "",
            }
        )
        for feature, coef in zip(fit.features, fit.coefficients):
            rows.append(
                {
                    "form": fit.form,
                    "term": feature,
                    "coefficient": float(coef),
                    "abs_coefficient": abs(float(coef)),
                    "ridge_alpha": fit.ridge_alpha,
                    "feature_mean": fit.feature_means[feature],
                    "feature_std": fit.feature_stds[feature],
                }
            )
    return pd.DataFrame(rows)


def validation_tables(lambda_values: pd.DataFrame):
    validation_rows = []
    orthogonality_rows = []
    for form, form_df in lambda_values.groupby("form"):
        groups = [("ALL", form_df), *list(form_df.groupby("task"))]
        for task, group in groups:
            lam = group["lambda_weight"]
            contribution = group["delta_accuracy"]
            quality = group["quality_score"]
            validation_rows.append(
                {
                    "form": form,
                    "task": task,
                    "n": len(group),
                    "lambda_mean": float(lam.mean()),
                    "lambda_std": float(lam.std(ddof=0)),
                    "lambda_min": float(lam.min()),
                    "lambda_max": float(lam.max()),
                    "lambda_cv": float(lam.std(ddof=0) / max(abs(lam.mean()), EPS)),
                    "lambda_delta_pearson": pearson(lam, contribution),
                    "lambda_delta_spearman": spearman(lam, contribution),
                }
            )
            orthogonality_rows.append(
                {
                    "form": form,
                    "task": task,
                    "n": len(group),
                    "lambda_quality_pearson": pearson(lam, quality),
                    "lambda_quality_spearman": spearman(lam, quality),
                    "quality_delta_pearson": pearson(quality, contribution),
                    "quality_delta_spearman": spearman(quality, contribution),
                    "mean_effective_quality": float(group["effective_quality_score"].mean()),
                }
            )
    return pd.DataFrame(validation_rows), pd.DataFrame(orthogonality_rows)


def _model_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > EPS else 0.0
    return rmse, mae, r2


def leave_one_task_out(df: pd.DataFrame):
    rows = []
    for held_out in sorted(df["task"].unique()):
        train = df[df["task"] != held_out].reset_index(drop=True)
        test = df[df["task"] == held_out].reset_index(drop=True)

        fit_a = fit_form_a(train)
        pred_a = predict_delta_accuracy(test, fit_a)
        rmse, mae, r2 = _model_metrics(test[TARGET], pred_a)
        rows.append(
            {
                "form": "form_a",
                "held_out_task": held_out,
                "ridge_alpha": "",
                "n_train": len(train),
                "n_test": len(test),
                "rmse": rmse,
                "mae": mae,
                "r_squared": r2,
                "pearson": pearson(pred_a, test[TARGET]),
                "spearman": spearman(pred_a, test[TARGET]),
            }
        )

        for alpha in RIDGE_ALPHAS:
            fit_b = fit_form_b(train, alpha)
            pred_b = predict_delta_accuracy(test, fit_b)
            rmse, mae, r2 = _model_metrics(test[TARGET], pred_b)
            rows.append(
                {
                    "form": "form_b",
                    "held_out_task": held_out,
                    "ridge_alpha": alpha,
                    "n_train": len(train),
                    "n_test": len(test),
                    "rmse": rmse,
                    "mae": mae,
                    "r_squared": r2,
                    "pearson": pearson(pred_b, test[TARGET]),
                    "spearman": spearman(pred_b, test[TARGET]),
                }
            )
    cv = pd.DataFrame(rows)
    ridge_rows = cv[cv["form"] == "form_b"].copy()
    mean_rmse = ridge_rows.groupby("ridge_alpha")["rmse"].mean()
    selected_alpha = float(mean_rmse.idxmin())
    return cv, selected_alpha
