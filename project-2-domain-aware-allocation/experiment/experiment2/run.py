"""Project 2 Experiment 2: calibrate a domain-aware aggregation lambda."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT2_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_CODE_ROOT = PROJECT2_ROOT / "experiment"
for path in (PROJECT2_ROOT, EXPERIMENT_CODE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiment2.lambda_calibration import (  # noqa: E402
    EXTENDED_RIDGE_ALPHAS,
    FORM_A_FEATURES,
    FORM_B_FEATURES,
    GROUP_COLS,
    RIDGE_ALPHAS,
    attach_lambda_values,
    calibrate_lambda_scales,
    coefficient_table,
    fit_form_a,
    fit_form_b,
    prepare_measurements,
    predict_standardized_score,
    ridge_alpha_grid,
    validation_tables,
)
from experiment2.form_c import (  # noqa: E402
    FORM_C_EXPLORATORY_STATUS,
    FORM_C_FEATURES,
    FORM_C_METHOD_LABEL,
    FORM_C_TARGET_TRANSFORMATION,
    fit_form_c,
    form_c_zero_variance_summary,
    predict_form_c_score,
)
from experiment2.calibration_bundle import (  # noqa: E402
    atomic_write_json,
    build_calibration_bundle,
    stable_payload_hash,
    write_calibration_bundle,
)
from experiment2.evaluation import (  # noqa: E402
    EvaluationConfig,
    alpha_evaluation_table,
    evaluation_table,
    support_decision_evaluation,
)
from experiment2.figures import save_evaluation_figures  # noqa: E402
from experiment2.lambda_aggregation import normalized_aggregation_weights  # noqa: E402
from experiment2.numeric_validation import validate_experiment1_numeric_inputs  # noqa: E402
from experiment2.provenance import (  # noqa: E402
    load_json_without_duplicate_keys,
    validate_experiment1_measurement_inputs,
)
from experiment2.reporting import build_evaluation_report  # noqa: E402
from framework.utils import (  # noqa: E402
    ensure_disjoint_directory,
    ensure_not_cleanup_parent,
    environment_manifest,
    prepare_output_directory,
)


OUTPUT_ROOT = PROJECT2_ROOT / "outputs"
EXP1_DIR = PROJECT2_ROOT / "outputs" / "exp1"
OUTPUT_DIR = PROJECT2_ROOT / "outputs" / "exp2"
LAMBDA_VALUE_COLUMNS = [
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
    "form",
    "methodology_label",
    "exploratory_status",
    "gamma",
    "target_lambda_cv",
    "achieved_lambda_cv",
    "raw_lambda_score",
    "predicted_delta_accuracy",
    "relative_contribution_target",
    "lambda_weight",
    "effective_quality_score",
]
COEFFICIENT_COLUMNS = [
    "form",
    "term",
    "coefficient",
    "abs_coefficient",
    "ridge_alpha",
    "feature_mean",
    "feature_std",
    "methodology_label",
    "feature_transformation",
    "target_transformation",
    "exploratory_status",
]
VALIDATION_COLUMNS = [
    "form",
    "task",
    "n",
    "lambda_mean",
    "lambda_std",
    "lambda_min",
    "lambda_max",
    "lambda_cv",
    "lambda_delta_pearson",
    "lambda_delta_spearman",
]
ORTHOGONALITY_COLUMNS = [
    "form",
    "task",
    "n",
    "lambda_quality_pearson",
    "lambda_quality_spearman",
    "quality_delta_pearson",
    "quality_delta_spearman",
    "mean_effective_quality",
]
EVALUATION_METRIC_COLUMNS = [
    "form",
    "scope",
    "scope_value",
    "n",
    "rmse",
    "mae",
    "r_squared",
    "pearson",
    "spearman",
    "pairwise_ranking_accuracy",
    "kendall_tau",
    "permutation_p_value",
]


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required Experiment 1 output is missing: {path}")
    return pd.read_csv(path)


def _read_required_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required manifest is missing: {path}")
    return load_json_without_duplicate_keys(path)


def _write_experiment2_dataset_manifest(
    output_dir: Path,
    source_manifest_path: Path,
    source_dataset_manifest: dict,
) -> dict:
    manifest = {
        "experiment": "Experiment 2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_dataset_loading": "inherited_from_experiment1_measurements",
        "source_dataset_manifest_file": str(source_manifest_path),
        "source_dataset_manifest": source_dataset_manifest,
    }
    atomic_write_json(output_dir / "dataset_manifest.json", manifest)
    return manifest


def _atomic_dataframe_to_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _svg_scatter(path: Path, df: pd.DataFrame, x_col: str, y_col: str, title: str):
    width, height = 760, 480
    margin = 60
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    if len(x) == 0:
        return
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    if abs(x_max - x_min) < 1e-12:
        x_max = x_min + 1.0
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1.0

    def sx(value):
        return margin + (value - x_min) / (x_max - x_min) * (width - 2 * margin)

    def sy(value):
        return height - margin - (value - y_min) / (y_max - y_min) * (height - 2 * margin)

    colors = {"form_a": "#1f77b4", "form_b": "#d62728"}
    points = []
    for _, row in df.iterrows():
        color = colors.get(row.get("form", ""), "#333333")
        points.append(
            f'<circle cx="{sx(float(row[x_col])):.2f}" cy="{sy(float(row[y_col])):.2f}" '
            f'r="4" fill="{color}" fill-opacity="0.68" />'
        )
    content = "\n".join(points)
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="28" text-anchor="middle" font-size="20" font-family="Arial">{title}</text>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#222"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#222"/>
<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-size="13" font-family="Arial">{x_col}</text>
<text x="18" y="{height / 2}" text-anchor="middle" transform="rotate(-90 18 {height / 2})" font-size="13" font-family="Arial">{y_col}</text>
<text x="{margin}" y="{height - margin + 20}" font-size="11" font-family="Arial">{x_min:.3g}</text>
<text x="{width - margin}" y="{height - margin + 20}" text-anchor="end" font-size="11" font-family="Arial">{x_max:.3g}</text>
<text x="{margin - 8}" y="{height - margin}" text-anchor="end" font-size="11" font-family="Arial">{y_min:.3g}</text>
<text x="{margin - 8}" y="{margin}" text-anchor="end" font-size="11" font-family="Arial">{y_max:.3g}</text>
{content}
</svg>
""",
        encoding="utf-8",
    )


def _svg_histogram(path: Path, df: pd.DataFrame):
    width, height = 760, 480
    margin = 60
    forms = ["form_a", "form_b"]
    colors = {"form_a": "#1f77b4", "form_b": "#d62728"}
    bins = np.linspace(0.5, 1.5, 21)
    max_count = 1
    histograms = {}
    for form in forms:
        values = df[df["form"] == form]["lambda_weight"].to_numpy(dtype=float)
        counts, _ = np.histogram(values, bins=bins)
        histograms[form] = counts
        max_count = max(max_count, int(counts.max()))

    bars = []
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    bin_w = plot_w / (len(bins) - 1)
    for bin_id in range(len(bins) - 1):
        for offset, form in enumerate(forms):
            count = histograms[form][bin_id]
            bar_w = bin_w * 0.38
            x = margin + bin_id * bin_w + offset * bar_w
            h = count / max_count * plot_h
            y = height - margin - h
            bars.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" '
                f'fill="{colors[form]}" fill-opacity="0.72"/>'
            )
    content = "\n".join(bars)
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="28" text-anchor="middle" font-size="20" font-family="Arial">Lambda distribution</text>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#222"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#222"/>
<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-size="13" font-family="Arial">lambda_weight</text>
<text x="18" y="{height / 2}" text-anchor="middle" transform="rotate(-90 18 {height / 2})" font-size="13" font-family="Arial">count</text>
<rect x="{width - 170}" y="55" width="12" height="12" fill="#1f77b4"/><text x="{width - 152}" y="66" font-size="12" font-family="Arial">Form A</text>
<rect x="{width - 170}" y="75" width="12" height="12" fill="#d62728"/><text x="{width - 152}" y="86" font-size="12" font-family="Arial">Form B</text>
{content}
</svg>
""",
        encoding="utf-8",
    )


def save_figures(lambda_values: pd.DataFrame, figure_dir: Path):
    figure_dir.mkdir(parents=True, exist_ok=True)
    _svg_histogram(figure_dir / "lambda_distribution.svg", lambda_values)
    _svg_scatter(
        figure_dir / "lambda_vs_contribution.svg",
        lambda_values,
        "lambda_weight",
        "delta_accuracy",
        "Lambda vs contribution",
    )
    _svg_scatter(
        figure_dir / "lambda_vs_quality.svg",
        lambda_values,
        "lambda_weight",
        "quality_score",
        "Lambda vs q",
    )
    wide = lambda_values.pivot_table(
        index=["task", "round", "client_id"],
        columns="form",
        values="lambda_weight",
        aggfunc="first",
    ).reset_index()
    if {"form_a", "form_b"}.issubset(wide.columns):
        _svg_scatter(
            figure_dir / "form_a_vs_form_b.svg",
            wide.rename(columns={"form_a": "lambda_form_a", "form_b": "lambda_form_b"}),
            "lambda_form_a",
            "lambda_form_b",
            "Form A vs Form B",
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp1-dir", type=Path, default=EXP1_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--ridge-alphas",
        nargs="*",
        type=float,
        default=None,
        help=(
            "Optional positive Ridge alpha grid. Defaults to the existing "
            f"grid: {RIDGE_ALPHAS}."
        ),
    )
    parser.add_argument(
        "--include-extended-ridge-alphas",
        action="store_true",
        help=f"Append prepared larger alpha candidates: {EXTENDED_RIDGE_ALPHAS}.",
    )
    parser.add_argument(
        "--ranking-permutations",
        type=int,
        default=1000,
        help="Number of permutations for ranking significance tests.",
    )
    parser.add_argument(
        "--ranking-permutation-seed",
        type=int,
        default=42,
        help="Seed for ranking permutation tests.",
    )
    parser.add_argument(
        "--allow-synthetic-source",
        action="store_true",
        help=(
            "Retained for synthetic audits; scientific calibration still rejects "
            "synthetic measurement provenance."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Clean the Experiment 2 output directory before running. By default "
            "a non-empty output directory is rejected."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_disjoint_directory(
        args.output_dir,
        args.exp1_dir,
        output_label="Experiment 2 output directory",
        protected_label="Experiment 1 input directory",
    )
    ensure_disjoint_directory(
        args.output_dir,
        EXP1_DIR,
        output_label="Experiment 2 output directory",
        protected_label="default Experiment 1 output directory",
    )
    ensure_not_cleanup_parent(
        args.output_dir,
        OUTPUT_ROOT,
        output_label="Experiment 2 output directory",
        protected_label="shared Project 2 outputs directory",
    )

    measurements = _read_required_csv(args.exp1_dir / "per_round_client_measurements.csv")
    exp1_manifest_path = args.exp1_dir / "manifest.json"
    exp1_manifest = _read_required_json(exp1_manifest_path)
    exp1_dataset_manifest_path = args.exp1_dir / "dataset_manifest.json"
    exp1_dataset_manifest = _read_required_json(exp1_dataset_manifest_path)
    correlations = _read_required_csv(args.exp1_dir / "signal_contribution_correlations.csv")
    regressions = _read_required_csv(args.exp1_dir / "controlled_regression.csv")
    label_distribution = _read_required_csv(args.exp1_dir / "label_distribution_summary.csv")

    normalized_inputs = validate_experiment1_measurement_inputs(
        measurements=measurements,
        label_distribution=label_distribution,
        correlations=correlations,
        regressions=regressions,
        dataset_manifest=exp1_dataset_manifest,
        measurement_label=args.exp1_dir / "per_round_client_measurements.csv",
        label_distribution_label=args.exp1_dir / "label_distribution_summary.csv",
        correlations_label=args.exp1_dir / "signal_contribution_correlations.csv",
        regressions_label=args.exp1_dir / "controlled_regression.csv",
    )
    measurements = normalized_inputs["measurements"]
    label_distribution = normalized_inputs["label_distribution"]
    correlations = normalized_inputs["correlations"]
    regressions = normalized_inputs["regressions"]
    del label_distribution

    numeric_inputs = validate_experiment1_numeric_inputs(
        measurements=measurements,
        label_distribution=normalized_inputs["label_distribution"],
        correlations=correlations,
        regressions=regressions,
        measurement_label=args.exp1_dir / "per_round_client_measurements.csv",
        label_distribution_label=args.exp1_dir / "label_distribution_summary.csv",
        correlations_label=args.exp1_dir / "signal_contribution_correlations.csv",
        regressions_label=args.exp1_dir / "controlled_regression.csv",
    )
    measurements = numeric_inputs["measurements"]
    correlations = numeric_inputs["correlations"]
    regressions = numeric_inputs["regressions"]

    args.output_dir = prepare_output_directory(
        args.output_dir,
        overwrite=args.overwrite,
        experiment_name="Experiment 2",
        allowed_cleanup_root=OUTPUT_DIR,
        repository_root=PROJECT2_ROOT.parent,
        project_root=PROJECT2_ROOT,
        shared_outputs_root=OUTPUT_ROOT,
    )
    figure_dir = args.output_dir / "figures"

    dataset_manifest = _write_experiment2_dataset_manifest(
        output_dir=args.output_dir,
        source_manifest_path=exp1_dataset_manifest_path,
        source_dataset_manifest=exp1_dataset_manifest,
    )

    df = prepare_measurements(measurements)
    experiment1_run_id = exp1_manifest.get("run_id") or stable_payload_hash(
        {
            "experiment1_manifest": exp1_manifest,
            "experiment1_dataset_manifest": exp1_dataset_manifest,
        }
    )
    ridge_alphas = ridge_alpha_grid(
        include_extended=args.include_extended_ridge_alphas,
        custom_alphas=args.ridge_alphas,
    )
    evaluation_config = EvaluationConfig(
        permutation_seed=args.ranking_permutation_seed,
        permutations=args.ranking_permutations,
    )
    cv, form_support, support_decisions = support_decision_evaluation(
        df,
        ridge_alphas=ridge_alphas,
        config=evaluation_config,
        source_run_identity=str(experiment1_run_id),
    )
    selected_alpha = support_decisions["form_b"].selected_alpha
    selected_form_c_alpha = support_decisions["form_c"].selected_alpha
    supported_fits = []
    if support_decisions["form_a"].supported:
        supported_fits.append(fit_form_a(df))
    if support_decisions["form_b"].supported:
        supported_fits.append(fit_form_b(df, float(selected_alpha)))
    if support_decisions["form_c"].supported:
        supported_fits.append(fit_form_c(df, float(selected_form_c_alpha)))

    scores = {}
    for fit in supported_fits:
        if fit.form == "form_c":
            scores[fit.form] = predict_form_c_score(df, fit)
        else:
            scores[fit.form] = predict_standardized_score(df, fit)
    lambda_calibrations = (
        calibrate_lambda_scales(df, scores) if supported_fits else {}
    )
    lambda_values = (
        attach_lambda_values(df, supported_fits, lambda_calibrations)
        if supported_fits
        else pd.DataFrame(columns=LAMBDA_VALUE_COLUMNS)
    )
    coefficients = (
        coefficient_table(supported_fits)
        if supported_fits
        else pd.DataFrame(columns=COEFFICIENT_COLUMNS)
    )
    if lambda_values.empty:
        validation = pd.DataFrame(columns=VALIDATION_COLUMNS)
        orthogonality = pd.DataFrame(columns=ORTHOGONALITY_COLUMNS)
        evaluation_metrics = pd.DataFrame(columns=EVALUATION_METRIC_COLUMNS)
    else:
        validation, orthogonality = validation_tables(lambda_values)
        evaluation_metrics = evaluation_table(lambda_values, config=evaluation_config)
    alpha_metrics = alpha_evaluation_table(cv, config=evaluation_config)
    ranking_columns = [
        "form",
        "scope",
        "scope_value",
        "n",
        "spearman",
        "kendall_tau",
        "permutation_p_value",
    ]
    ranking_significance = evaluation_metrics[ranking_columns].copy()

    _atomic_dataframe_to_csv(lambda_values, args.output_dir / "lambda_values.csv")
    _atomic_dataframe_to_csv(validation, args.output_dir / "lambda_validation.csv")
    _atomic_dataframe_to_csv(orthogonality, args.output_dir / "orthogonality_report.csv")
    _atomic_dataframe_to_csv(cv, args.output_dir / "cross_validation.csv")
    _atomic_dataframe_to_csv(form_support, args.output_dir / "form_support_status.csv")
    _atomic_dataframe_to_csv(coefficients, args.output_dir / "fitted_coefficients.csv")
    _atomic_dataframe_to_csv(evaluation_metrics, args.output_dir / "evaluation_metrics.csv")
    _atomic_dataframe_to_csv(alpha_metrics, args.output_dir / "alpha_evaluation.csv")
    _atomic_dataframe_to_csv(ranking_significance, args.output_dir / "ranking_significance.csv")

    figure_files = []
    if not lambda_values.empty:
        save_figures(lambda_values, figure_dir)
        figure_files.extend(
            [
                "figures/lambda_distribution.svg",
                "figures/lambda_vs_contribution.svg",
                "figures/lambda_vs_quality.svg",
            ]
        )
        if {"form_a", "form_b"}.issubset(set(lambda_values["form"])):
            figure_files.append("figures/form_a_vs_form_b.svg")
        figure_files.extend(
            save_evaluation_figures(
                figure_dir,
                evaluation_metrics=evaluation_metrics,
                alpha_metrics=alpha_metrics,
            )
        )
    build_evaluation_report(
        output_dir=args.output_dir,
        correlations=correlations,
        regressions=regressions,
        coefficients=coefficients,
        validation=validation,
        orthogonality=orthogonality,
        cv=cv,
        evaluation_metrics=evaluation_metrics,
        alpha_metrics=alpha_metrics,
        form_support=form_support,
        selected_alpha=selected_alpha,
        lambda_calibrations=lambda_calibrations,
        evaluation_config={
            "permutations": args.ranking_permutations,
            "permutation_seed": args.ranking_permutation_seed,
        },
        figure_files=figure_files,
    )

    experiment2_run_id = stable_payload_hash(
        {
            "source_experiment1_run_id": experiment1_run_id,
            "task_set": list(df["task"].drop_duplicates()),
            "ridge_alpha_grid": ridge_alphas,
            "selected_ridge_alpha": selected_alpha,
            "selected_form_c_ridge_alpha": selected_form_c_alpha,
            "form_support": form_support.to_dict(orient="records"),
            "supported_treatment_arms": [fit.form for fit in supported_fits],
            "lambda_calibration": lambda_calibrations,
        }
    )
    calibration_bundle = None
    if supported_fits:
        calibration_bundle = build_calibration_bundle(
            measurements=df,
            exp1_manifest=exp1_manifest,
            exp1_dataset_manifest=exp1_dataset_manifest,
            fits=supported_fits,
            lambda_calibrations=lambda_calibrations,
            experiment1_run_id=experiment1_run_id,
            experiment2_run_id=experiment2_run_id,
            form_support=form_support,
        )
        write_calibration_bundle(args.output_dir / "calibration_bundle.json", calibration_bundle)

    supported_treatment_arms = [fit.form for fit in supported_fits]
    outputs = [
        "lambda_values.csv",
        "lambda_validation.csv",
        "orthogonality_report.csv",
        "cross_validation.csv",
        "form_support_status.csv",
        "fitted_coefficients.csv",
        "evaluation_metrics.csv",
        "alpha_evaluation.csv",
        "ranking_significance.csv",
        "comparison_report.md",
    ]
    if calibration_bundle is not None:
        outputs.append("calibration_bundle.json")
    if figure_files:
        outputs.append("figures/")

    manifest = {
        "project": "Project 2",
        "experiment": "Experiment 2",
        "status": "complete" if supported_fits else "complete_negative_no_supported_forms",
        "source_experiment": str(args.exp1_dir),
        "source_experiment1_run_id": experiment1_run_id,
        "experiment2_run_id": experiment2_run_id,
        "output_dir": str(args.output_dir),
        "dataset_manifest_file": "dataset_manifest.json",
        "calibration_bundle_file": "calibration_bundle.json" if calibration_bundle else None,
        "calibration_bundle_schema_version": calibration_bundle["schema_version"] if calibration_bundle else None,
        "dataset_manifest": dataset_manifest,
        "source_dataset_provenance": exp1_manifest.get("dataset_provenance", {}),
        "is_synthetic_present": bool(df["is_synthetic"].astype(bool).any()),
        "environment": environment_manifest(),
        "form_a_features": FORM_A_FEATURES,
        "form_b_features": FORM_B_FEATURES,
        "form_c_features": FORM_C_FEATURES,
        "form_c_methodology": {
            "label": FORM_C_METHOD_LABEL,
            "exploratory_status": FORM_C_EXPLORATORY_STATUS,
            "feature_transformation": "within task-round population z-score",
            "target_transformation": FORM_C_TARGET_TRANSFORMATION,
            "intercept_policy": "fit_intercept_false_because_features_and_target_are_group_centered",
            "zero_variance_behavior": "map zero-variance predictor or target groups to zero without dropping rows",
            "zero_variance_summary": form_c_zero_variance_summary(df),
        },
        "ridge_alpha_grid": ridge_alphas,
        "extended_ridge_alpha_candidates": EXTENDED_RIDGE_ALPHAS,
        "selected_ridge_alpha": selected_alpha,
        "selected_form_c_ridge_alpha": selected_form_c_alpha,
        "selected_ridge_alpha_on_boundary": support_decisions["form_b"].alpha_boundary_status in {"minimum", "maximum"},
        "selected_form_c_ridge_alpha_on_boundary": support_decisions["form_c"].alpha_boundary_status in {"minimum", "maximum"},
        "alpha_selection_rule": "minimum mean leave-one-task-out RMSE, accepted only when interior",
        "primary_model_validity_gate": {
            "null_model": "fold-safe train-fold intercept-only target mean",
            "primary_metric": "raw mean leave-one-task-out RMSE",
            "support_rule": "model RMSE must be strictly lower than null by more than numerical tolerance; Ridge alpha must be interior",
            "null_improvement_tolerance": 1e-9,
        },
        "form_c_validity_gate": {
            "null_model": "zero prediction for within-task-round centered target",
            "primary_metric": "mean group-normalized leave-one-task-out RMSE with equal task weighting",
            "support_rule": "Form C normalized RMSE must be strictly lower than zero-null by more than numerical tolerance; selected Ridge alpha must be interior; predictive orientation must not be reversed",
            "ranking_metrics_secondary_only": True,
        },
        "form_support": form_support.to_dict(orient="records"),
        "supported_treatment_arms": supported_treatment_arms,
        "experiment3_eligible_arms": ["baseline", *supported_treatment_arms],
        "ranking_metrics_not_used_for_selection": True,
        "evaluation": {
            "regression_metrics": ["rmse", "mae", "r_squared", "pearson"],
            "ranking_metrics": [
                "spearman",
                "pairwise_ranking_accuracy",
                "kendall_tau",
            ],
            "statistical_tests": ["spearman_permutation_p_value"],
            "permutations": args.ranking_permutations,
            "permutation_seed": args.ranking_permutation_seed,
        },
        "lambda_calibration": lambda_calibrations,
        "normalization": {
            "context": GROUP_COLS,
            "positive_transform": "exp(scale * centered_score)",
            "clip": [0.5, 1.5],
            "renormalize_context_mean": 1.0,
        },
        "aggregation_rule": "Weight = w * q * lambda",
        "disabled_behavior": "lambda_weights=None preserves Weight = w * q",
        "outputs": outputs,
        "sanity_check_weights": normalized_aggregation_weights(
            samples=[1, 1, 1],
            quality_scores=[1, 2, 3],
            lambda_weights=None,
        ),
        "sanity_check_weights_with_lambda": normalized_aggregation_weights(
            samples=[1, 1, 1],
            quality_scores=[1, 2, 3],
            lambda_weights=[1, 1, 1],
        ),
    }
    atomic_write_json(args.output_dir / "manifest.json", manifest)

    print("=== Experiment 2 Complete ===")
    print(f"Rows used: {len(df)}")
    print(f"Selected ridge alpha: {selected_alpha}")
    print(f"Selected Form C ridge alpha: {selected_form_c_alpha}")
    print(f"Supported treatment arms: {supported_treatment_arms}")
    print(f"Lambda calibration: {lambda_calibrations}")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
