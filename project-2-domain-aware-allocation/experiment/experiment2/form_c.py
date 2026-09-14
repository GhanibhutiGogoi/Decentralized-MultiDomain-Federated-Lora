"""Shared Form C within-round relative-contribution calibration utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from experiment2.lambda_calibration import (
    EPS,
    FORM_B_FEATURES,
    GROUP_COLS,
    TARGET,
    LinearFit,
)


FORM_C_FEATURES = list(FORM_B_FEATURES)
FORM_C_METHOD_LABEL = "Form C - within-round relative-contribution calibration"
FORM_C_TARGET_TRANSFORMATION = "within_task_round_zscore_delta_accuracy"
FORM_C_FEATURE_TRANSFORMATION = "within_task_round_population_zscore"
FORM_C_EXPLORATORY_STATUS = "post_hoc_exploratory"


@dataclass(frozen=True)
class FormCTransformResult:
    """Transformed Form C table and zero-variance diagnostics."""

    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    target_column: str
    feature_zero_variance_counts: Mapping[str, int]
    target_zero_variance_group_count: int
    group_count: int


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite.")
    return parsed


def validate_form_c_feature_order(features: Sequence[str]) -> tuple[str, ...]:
    """Require Form C to reuse the deployable Form B predictor order."""
    feature_tuple = tuple(str(feature) for feature in features)
    expected = tuple(FORM_C_FEATURES)
    if feature_tuple != expected:
        raise ValueError(f"Form C feature order must exactly equal {list(expected)!r}.")
    return feature_tuple


def transform_form_c_context(
    df: pd.DataFrame,
    *,
    features: Sequence[str] = FORM_C_FEATURES,
    include_target: bool = True,
    group_cols: Sequence[str] = GROUP_COLS,
    target_col: str = TARGET,
) -> FormCTransformResult:
    """Apply Form C within-group population z-scoring without dropping rows.

    The same routine is used for Experiment 2 fitting/evaluation and
    Experiment 3 inference. Predictors are transformed only from current
    aggregation-context client features. The target transformation is optional
    and is used only for calibration/evaluation.
    """
    feature_tuple = validate_form_c_feature_order(features)
    group_tuple = tuple(str(col) for col in group_cols)
    required = [*group_tuple, *feature_tuple]
    if include_target:
        required.append(target_col)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Form C input is missing columns: {missing}.")

    out = df.copy(deep=True).reset_index(drop=True)
    if out.empty:
        raise ValueError("Form C transformation requires at least one row.")
    if out[list(group_tuple)].isna().any().any():
        raise ValueError("Form C group identifiers must be complete.")

    transformed_features = tuple(f"form_c__{feature}" for feature in feature_tuple)
    feature_zero_counts = {feature: 0 for feature in feature_tuple}
    target_zero_count = 0
    group_count = 0

    for _, indices in out.groupby(list(group_tuple), sort=False).groups.items():
        positions = list(indices)
        group_count += 1
        for feature, transformed in zip(feature_tuple, transformed_features):
            values = out.loc[positions, feature].map(
                lambda value, feature=feature: _finite_float(value, feature)
            ).to_numpy(dtype=float)
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            if std <= EPS:
                out.loc[positions, transformed] = 0.0
                feature_zero_counts[feature] += 1
            else:
                out.loc[positions, transformed] = (values - mean) / std

        if include_target:
            y = out.loc[positions, target_col].map(
                lambda value: _finite_float(value, target_col)
            ).to_numpy(dtype=float)
            mean_y = float(y.mean())
            std_y = float(y.std(ddof=0))
            if std_y <= EPS:
                out.loc[positions, "form_c__target"] = 0.0
                target_zero_count += 1
            else:
                out.loc[positions, "form_c__target"] = (y - mean_y) / std_y

    target_name = "form_c__target" if include_target else ""
    numeric_columns = list(transformed_features) + ([target_name] if target_name else [])
    for column in numeric_columns:
        values = out[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Form C transformed column {column!r} is not finite.")

    return FormCTransformResult(
        frame=out,
        feature_columns=transformed_features,
        target_column=target_name,
        feature_zero_variance_counts=feature_zero_counts,
        target_zero_variance_group_count=target_zero_count,
        group_count=group_count,
    )


def fit_form_c(df: pd.DataFrame, ridge_alpha: float) -> LinearFit:
    """Fit Form C Ridge on within-task-round relative predictors and target."""
    alpha = _finite_float(ridge_alpha, "form_c.ridge_alpha")
    if alpha <= 0.0:
        raise ValueError("form_c.ridge_alpha must be positive.")
    transformed = transform_form_c_context(df, include_target=True)
    x = transformed.frame[list(transformed.feature_columns)].to_numpy(dtype=float)
    y = transformed.frame[transformed.target_column].to_numpy(dtype=float)
    xtx = x.T @ x
    xty = x.T @ y
    coef = np.linalg.solve(xtx + alpha * np.eye(x.shape[1]), xty)
    if not np.all(np.isfinite(coef)):
        raise ValueError("Form C coefficients must be finite.")
    return LinearFit(
        form="form_c",
        features=list(FORM_C_FEATURES),
        coefficients=coef,
        intercept=0.0,
        feature_means={feature: 0.0 for feature in FORM_C_FEATURES},
        feature_stds={feature: 1.0 for feature in FORM_C_FEATURES},
        target_mean=0.0,
        target_std=1.0,
        ridge_alpha=float(alpha),
        methodology_label=FORM_C_METHOD_LABEL,
        feature_transformation=FORM_C_FEATURE_TRANSFORMATION,
        target_transformation=FORM_C_TARGET_TRANSFORMATION,
        exploratory_status=FORM_C_EXPLORATORY_STATUS,
    )


def predict_form_c_score(df: pd.DataFrame, fit: LinearFit) -> np.ndarray:
    """Predict Form C relative scores using current group-only features."""
    validate_form_c_feature_order(fit.features)
    transformed = transform_form_c_context(df, features=fit.features, include_target=False)
    x = transformed.frame[list(transformed.feature_columns)].to_numpy(dtype=float)
    score = float(fit.intercept) + x @ np.asarray(fit.coefficients, dtype=float)
    if not np.all(np.isfinite(score)):
        raise ValueError("Form C scores must be finite.")
    return score.astype(float)


def form_c_zero_variance_summary(df: pd.DataFrame) -> dict[str, object]:
    """Return zero-variance diagnostics for a full calibration table."""
    transformed = transform_form_c_context(df, include_target=True)
    return {
        "group_count": int(transformed.group_count),
        "feature_zero_variance_counts": dict(transformed.feature_zero_variance_counts),
        "target_zero_variance_group_count": int(
            transformed.target_zero_variance_group_count
        ),
    }


def coefficient_norm(coefficients: Iterable[float]) -> float:
    values = np.asarray(list(coefficients), dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.linalg.norm(values))
