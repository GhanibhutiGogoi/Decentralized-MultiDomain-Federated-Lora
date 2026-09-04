"""Statistical analysis for Project 2 Experiment 1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PREDICTORS = [
    "js_to_global",
    "kl_to_global",
    "update_cosine_distance_to_mean",
    "update_l2_distance_to_mean",
]

CONTROLS = ["adaptive_rank", "local_loss"]


def _source_is_synthetic(measurements: pd.DataFrame) -> bool:
    if measurements.empty:
        return False
    if "is_synthetic" not in measurements:
        raise ValueError(
            "Experiment 1 analysis requires measurement-level 'is_synthetic' "
            "provenance before writing derived artifacts."
        )
    parsed = measurements["is_synthetic"].map(
        {
            True: True,
            False: False,
            "True": True,
            "False": False,
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            1: True,
            0: False,
        }
    )
    if parsed.isna().any():
        raise ValueError(
            "Experiment 1 analysis found missing or ambiguous 'is_synthetic' "
            "measurement provenance."
        )
    return bool(parsed.astype(bool).any())


def _standardize(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return series * 0.0
    return (series - series.mean()) / std


def _ols(y: np.ndarray, x: np.ndarray):
    x = np.column_stack([np.ones(len(x)), x])
    beta, residuals, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    y_hat = x @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    dof = max(len(y) - rank, 1)
    sigma2 = ss_res / dof
    cov = sigma2 * np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return beta, se, r2


def run_statistical_analysis(measurements: pd.DataFrame, output_dir: Path):
    """Save correlation and controlled-regression tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    is_synthetic = _source_is_synthetic(measurements)

    needed = ["delta_accuracy", *PREDICTORS, *CONTROLS]
    df = measurements.dropna(subset=[col for col in needed if col in measurements])
    rows = []
    reg_rows = []

    for predictor in PREDICTORS:
        if predictor not in df:
            continue
        pair = df[[predictor, "delta_accuracy"]].dropna()
        if len(pair) >= 2:
            rows.append(
                {
                    "predictor": predictor,
                    "is_synthetic": is_synthetic,
                    "pearson": pair[predictor].corr(pair["delta_accuracy"], method="pearson"),
                    "spearman": pair[predictor].corr(pair["delta_accuracy"], method="spearman"),
                    "n": len(pair),
                }
            )

        cols = [predictor, *[control for control in CONTROLS if control in df]]
        model_df = df[[*cols, "delta_accuracy"]].dropna()
        if len(model_df) < len(cols) + 2:
            continue
        y = _standardize(model_df["delta_accuracy"]).to_numpy(dtype=float)
        x = np.column_stack([
            _standardize(model_df[col]).to_numpy(dtype=float)
            for col in cols
        ])
        beta, se, r2 = _ols(y, x)
        for idx, col in enumerate(cols, start=1):
            reg_rows.append(
                {
                    "model_predictor": predictor,
                    "term": col,
                    "is_synthetic": is_synthetic,
                    "standardized_beta": beta[idx],
                    "standard_error": se[idx],
                    "r_squared": r2,
                    "n": len(model_df),
                }
            )

    corr_df = pd.DataFrame(rows)
    reg_df = pd.DataFrame(reg_rows)
    corr_df.to_csv(output_dir / "signal_contribution_correlations.csv", index=False)
    reg_df.to_csv(output_dir / "controlled_regression.csv", index=False)
    return corr_df, reg_df

