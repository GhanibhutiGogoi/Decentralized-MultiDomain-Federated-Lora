"""Neutral report generation for Experiment 2 evaluation outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiment2.form_c import FORM_C_FEATURES, FORM_C_METHOD_LABEL
from experiment2.lambda_calibration import FORM_A_FEATURES, FORM_B_FEATURES


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """Render a compact Markdown table without optional tabulate dependency."""
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.6g}")
        else:
            display[column] = display[column].astype(str)
    headers = list(display.columns)
    rows = display.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_evaluation_report(
    output_dir: Path,
    *,
    correlations: pd.DataFrame,
    regressions: pd.DataFrame,
    coefficients: pd.DataFrame,
    validation: pd.DataFrame,
    orthogonality: pd.DataFrame,
    cv: pd.DataFrame,
    evaluation_metrics: pd.DataFrame,
    alpha_metrics: pd.DataFrame,
    form_support: pd.DataFrame,
    selected_alpha: float,
    lambda_calibrations: dict[str, dict[str, float]],
    evaluation_config: dict,
    figure_files: list[str],
) -> None:
    """Write a report derived from outputs without scientific conclusions."""
    calibration_md = markdown_table(pd.DataFrame(
        [
            {"form": form, **values}
            for form, values in sorted(lambda_calibrations.items())
        ]
    ))
    if evaluation_metrics.empty:
        global_metrics = evaluation_metrics
        task_metrics = evaluation_metrics
    else:
        global_metrics = evaluation_metrics[
            (evaluation_metrics["scope"] == "all")
            & (evaluation_metrics["scope_value"] == "ALL")
        ]
        task_metrics = evaluation_metrics[evaluation_metrics["scope"] == "task"]
    validation_all = (
        validation[validation["task"] == "ALL"]
        if "task" in validation.columns
        else validation
    )
    orthogonality_all = (
        orthogonality[orthogonality["task"] == "ALL"]
        if "task" in orthogonality.columns
        else orthogonality
    )
    report = f"""# Experiment 2 Evaluation Report

## Scope

This report is generated from Experiment 2 output tables. It reports regression,
ranking, and statistical evaluation metrics without selecting a preferred lambda
form or changing the calibration method.

## Inputs

Experiment 1 signal tables:

{markdown_table(correlations)}

Experiment 1 controlled regressions:

{markdown_table(regressions)}

## Lambda Forms

Form A features: `{", ".join(FORM_A_FEATURES)}`.

Form B features: `{", ".join(FORM_B_FEATURES)}`.

Form C label: `{FORM_C_METHOD_LABEL}`.

Form C features: `{", ".join(FORM_C_FEATURES)}`.

Form C was introduced after observing cross-task target-scale mismatch in the
original A/B negative result. It is post-hoc and exploratory at calibration
time; Experiment 3 is required for independent confirmation if it is supported.

Selected Ridge alpha remains RMSE-based: `{selected_alpha}`.

## Null-Model Validity Gate

Experiment 2 reports a fold-safe intercept-only null comparator. A lambda form is
supported only when its raw mean leave-one-task-out RMSE is strictly lower than
the fold-safe null by more than the documented numerical tolerance; Ridge forms
must also select an interior alpha.

{markdown_table(form_support)}

## Gamma Calibration

{calibration_md}

## Coefficients

{markdown_table(coefficients)}

## Regression And Ranking Metrics

Global metrics:

{markdown_table(global_metrics)}

Per-task metrics:

{markdown_table(task_metrics)}

## Alpha Comparison

{markdown_table(alpha_metrics)}

## Lambda Validation

{markdown_table(validation_all)}

## Orthogonality Against q

{markdown_table(orthogonality_all)}

## Leave-One-Task-Out Details

{markdown_table(cv)}

## Statistical Tests

Permutation p-values are computed as one-sided tests for positive Spearman rank
association using `{evaluation_config["permutations"]}` permutations and seed
`{evaluation_config["permutation_seed"]}`. Pooled evaluations stratify
permutations by the configured aggregation context columns when those columns
are available.

## Figures

{markdown_table(pd.DataFrame({"file": figure_files}))}

## Decision Status

Forms A and B remain recorded under their original raw-RMSE/null evaluation.
Unsupported forms do not receive coefficients, gamma, lambda values, or
Experiment 3 treatment-arm eligibility. If the support table contains no
supported treatment form, Experiment 3 is blocked because no scientifically
runnable domain-aware calibration bundle is emitted.
"""
    (output_dir / "comparison_report.md").write_text(report, encoding="utf-8")
