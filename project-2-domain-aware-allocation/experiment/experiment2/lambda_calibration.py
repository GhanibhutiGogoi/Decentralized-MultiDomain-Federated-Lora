"""Calibration utilities for Experiment 2 domain-aware lambda."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from experiment2.numeric_validation import MEASUREMENT_SCHEMA, validate_numeric_table
from experiment2.provenance import normalize_measurement_provenance


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
EXTENDED_RIDGE_ALPHAS = [300.0, 500.0, 1000.0]
LAMBDA_MIN = 0.5
LAMBDA_MAX = 1.5
MAX_LAMBDA_CV = 0.20
LAMBDA_CV_FRACTION_OF_Q = 0.5
CV_TOLERANCE = 1e-9
LAMBDA_BOUND_TOLERANCE = 1e-10


class RidgeAlphaBoundaryError(RuntimeError):
    """Raised when RMSE selects an alpha on the tested grid boundary."""


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
    out = normalize_measurement_provenance(
        df,
        file_label="Experiment 2 measurements",
    )
    out = validate_numeric_table(
        out,
        MEASUREMENT_SCHEMA,
        artifact_label="Experiment 2 measurements",
    )
    out["log_update_l2"] = np.log1p(out["update_l2_distance_to_mean"].clip(lower=0.0))
    out["log_class_imbalance_ratio"] = np.log1p(
        out["class_imbalance_ratio"].clip(lower=0.0)
    )
    return out.reset_index(drop=True)


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


def ridge_alpha_grid(
    *,
    include_extended: bool = False,
    custom_alphas: Iterable[float] | None = None,
) -> list[float]:
    """Return the Ridge alpha grid without changing the default search."""
    if custom_alphas is not None:
        alphas = [float(alpha) for alpha in custom_alphas]
        if include_extended:
            alphas.extend(EXTENDED_RIDGE_ALPHAS)
    else:
        alphas = list(RIDGE_ALPHAS)
        if include_extended:
            alphas.extend(EXTENDED_RIDGE_ALPHAS)
    unique = sorted(set(alphas))
    if not unique or any(alpha <= 0 for alpha in unique):
        raise ValueError("Ridge alphas must be positive.")
    return unique


def enforce_ridge_alpha_not_on_boundary(
    selected_alpha: float,
    ridge_alphas: Iterable[float],
) -> None:
    """Fail when the selected alpha lies on the tested search boundary."""
    alphas = sorted({float(alpha) for alpha in ridge_alphas})
    if not alphas:
        return
    lower = alphas[0]
    upper = alphas[-1]
    if selected_alpha == lower or selected_alpha == upper:
        side = "maximum" if selected_alpha == upper else "minimum"
        raise RidgeAlphaBoundaryError(
            "Selected Ridge alpha "
            f"{selected_alpha:g} is the {side} tested value in {alphas}. "
            "The optimum may lie outside the tested range. Expand the Ridge "
            "alpha search space before publishing results. The search grid was "
            "not expanded automatically."
        )


def warn_if_ridge_alpha_on_boundary(
    selected_alpha: float,
    ridge_alphas: Iterable[float],
) -> None:
    """Backward-compatible boundary guard; now raises instead of warning."""
    enforce_ridge_alpha_not_on_boundary(selected_alpha, ridge_alphas)


def _clip_renormalize_to_mean_one(
    values: np.ndarray,
    lower: float = LAMBDA_MIN,
    upper: float = LAMBDA_MAX,
    max_iter: int = 1000,
    tolerance: float = LAMBDA_BOUND_TOLERANCE,
) -> np.ndarray:
    """Iteratively clip and renormalize until bounds and mean-one both hold."""
    lam = np.asarray(values, dtype=float)
    if lam.size == 0:
        return lam
    if lower > 1.0 or upper < 1.0:
        raise ValueError("Lambda bounds must contain 1.0 for mean-one feasibility.")
    if not np.all(np.isfinite(lam)) or np.any(lam <= 0):
        raise ValueError("Lambda normalization requires finite positive inputs.")

    lam = lam / max(float(lam.mean()), EPS)
    for _ in range(max_iter):
        previous = lam.copy()
        lam = np.clip(lam, lower, upper)
        lam = lam / max(float(lam.mean()), EPS)
        bounds_hold = (
            float(lam.min()) >= lower - tolerance
            and float(lam.max()) <= upper + tolerance
        )
        mean_holds = abs(float(lam.mean()) - 1.0) <= tolerance
        converged = float(np.max(np.abs(lam - previous))) <= tolerance
        if bounds_hold and mean_holds and converged:
            return lam

    raise RuntimeError(
        "Lambda clipping failed to converge to the documented invariant: "
        f"mean=1 and bounds=[{lower}, {upper}]."
    )


def _lambda_from_score(df: pd.DataFrame, score: np.ndarray, scale: float):
    work = df[GROUP_COLS].reset_index(drop=True).copy()
    work["score"] = score
    centered = work["score"] - work.groupby(GROUP_COLS)["score"].transform("mean")
    raw = np.exp(scale * centered.clip(lower=-20.0, upper=20.0))
    work["lambda_raw"] = raw
    lam = np.empty(len(work), dtype=float)
    raw_values = work["lambda_raw"].to_numpy(dtype=float)
    for _, indices in work.groupby(GROUP_COLS).groups.items():
        group_positions = np.asarray(indices, dtype=int)
        lam[group_positions] = _clip_renormalize_to_mean_one(
            raw_values[group_positions]
        )
    return lam


def _coefficient_of_variation(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(arr.std(ddof=0) / max(abs(arr.mean()), EPS))


def calibrate_lambda_scales(
    df: pd.DataFrame,
    scores_by_form: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Calibrate gamma independently for each lambda form.

    The target for each form is CV(lambda_form) <= min(0.5 * CV(q), 0.20).
    A form-specific binary search chooses the largest gamma satisfying that
    bound. Failure to satisfy the bound raises instead of silently accepting an
    invalid calibration.
    """
    q_cv = float(df[QUALITY].std(ddof=0) / max(abs(df[QUALITY].mean()), EPS))
    target_cv = min(LAMBDA_CV_FRACTION_OF_Q * q_cv, MAX_LAMBDA_CV)
    calibrations: dict[str, dict[str, float]] = {}

    for form, scores in scores_by_form.items():
        if target_cv <= EPS:
            gamma = 0.0
            achieved_cv = 0.0
        else:
            lo, hi = 0.0, 5.0
            for _ in range(40):
                mid = (lo + hi) / 2.0
                lam = _lambda_from_score(df, scores, mid)
                cv = _coefficient_of_variation(lam)
                if cv > target_cv:
                    hi = mid
                else:
                    lo = mid
            gamma = lo
            achieved_cv = _coefficient_of_variation(_lambda_from_score(df, scores, gamma))

        if achieved_cv > target_cv + CV_TOLERANCE:
            raise RuntimeError(
                f"Gamma calibration failed for {form}: achieved CV "
                f"{achieved_cv:.12g} exceeds target CV {target_cv:.12g}."
            )

        calibrations[form] = {
            "gamma": float(gamma),
            "quality_cv": float(q_cv),
            "target_cv": float(target_cv),
            "achieved_cv": float(achieved_cv),
        }

    return calibrations


def attach_lambda_values(
    df: pd.DataFrame,
    fits: list[LinearFit],
    lambda_calibrations: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows = []
    base_cols = [
        "task",
        "round",
        "client_id",
        "is_synthetic",
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
        gamma = lambda_calibrations[fit.form]["gamma"]
        lam = _lambda_from_score(df, score, gamma)
        pred = predict_delta_accuracy(df, fit)
        part = df[base_cols].copy()
        part["form"] = fit.form
        part["gamma"] = gamma
        part["target_lambda_cv"] = lambda_calibrations[fit.form]["target_cv"]
        part["achieved_lambda_cv"] = lambda_calibrations[fit.form]["achieved_cv"]
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


def leave_one_task_out(df: pd.DataFrame, ridge_alphas: Iterable[float] | None = None):
    rows = []
    ridge_alphas = ridge_alpha_grid(custom_alphas=ridge_alphas)
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

        for alpha in ridge_alphas:
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
    enforce_ridge_alpha_not_on_boundary(selected_alpha, ridge_alphas)
    return cv, selected_alpha
