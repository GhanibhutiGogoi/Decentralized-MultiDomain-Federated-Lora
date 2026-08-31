"""Project 2 Experiment 2: calibrate a domain-aware aggregation lambda."""

from __future__ import annotations

import argparse
import json
import sys
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
from experiment2.evaluation import (  # noqa: E402
    EvaluationConfig,
    alpha_evaluation_table,
    evaluation_table,
    leave_one_task_out_evaluation,
)
from experiment2.figures import save_evaluation_figures  # noqa: E402
from experiment2.lambda_aggregation import normalized_aggregation_weights  # noqa: E402
from experiment2.reporting import build_evaluation_report  # noqa: E402
from framework.utils import environment_manifest  # noqa: E402


EXP1_DIR = PROJECT2_ROOT / "outputs" / "exp1"
OUTPUT_DIR = PROJECT2_ROOT / "outputs" / "exp2"


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required Experiment 1 output is missing: {path}")
    return pd.read_csv(path)


def _read_required_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required manifest is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_contains_synthetic(dataset_manifest: dict) -> bool:
    datasets = dataset_manifest.get("datasets", {})
    return any(
        bool(record.get("synthetic", False))
        for record in datasets.values()
        if isinstance(record, dict)
    )


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
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


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
        help="Explicitly allow Experiment 1 measurements produced from synthetic data.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"

    measurements = _read_required_csv(args.exp1_dir / "per_round_client_measurements.csv")
    exp1_manifest_path = args.exp1_dir / "manifest.json"
    exp1_manifest = _read_required_json(exp1_manifest_path)
    exp1_dataset_manifest_path = args.exp1_dir / "dataset_manifest.json"
    exp1_dataset_manifest = _read_required_json(exp1_dataset_manifest_path)
    if _manifest_contains_synthetic(exp1_dataset_manifest) and not args.allow_synthetic_source:
        raise RuntimeError(
            "Experiment 2 source measurements include synthetic datasets. "
            "Real datasets are required by default; rerun Experiment 2 with "
            "--allow-synthetic-source only for an explicit synthetic audit."
        )
    dataset_manifest = _write_experiment2_dataset_manifest(
        output_dir=args.output_dir,
        source_manifest_path=exp1_dataset_manifest_path,
        source_dataset_manifest=exp1_dataset_manifest,
    )
    correlations = _read_required_csv(args.exp1_dir / "signal_contribution_correlations.csv")
    regressions = _read_required_csv(args.exp1_dir / "controlled_regression.csv")
    _read_required_csv(args.exp1_dir / "label_distribution_summary.csv")

    df = prepare_measurements(measurements)
    ridge_alphas = ridge_alpha_grid(
        include_extended=args.include_extended_ridge_alphas,
        custom_alphas=args.ridge_alphas,
    )
    evaluation_config = EvaluationConfig(
        permutation_seed=args.ranking_permutation_seed,
        permutations=args.ranking_permutations,
    )
    cv, selected_alpha = leave_one_task_out_evaluation(
        df,
        ridge_alphas=ridge_alphas,
        config=evaluation_config,
    )
    fit_a = fit_form_a(df)
    fit_b = fit_form_b(df, selected_alpha)
    scores = {
        "form_a": predict_standardized_score(df, fit_a),
        "form_b": predict_standardized_score(df, fit_b),
    }
    lambda_calibrations = calibrate_lambda_scales(df, scores)
    lambda_values = attach_lambda_values(df, [fit_a, fit_b], lambda_calibrations)
    coefficients = coefficient_table([fit_a, fit_b])
    validation, orthogonality = validation_tables(lambda_values)
    evaluation_metrics = evaluation_table(lambda_values, config=evaluation_config)
    alpha_metrics = alpha_evaluation_table(cv, config=evaluation_config)
    ranking_significance = evaluation_metrics[
        ["form", "scope", "scope_value", "n", "spearman", "kendall_tau", "permutation_p_value"]
    ].copy()

    lambda_values.to_csv(args.output_dir / "lambda_values.csv", index=False)
    validation.to_csv(args.output_dir / "lambda_validation.csv", index=False)
    orthogonality.to_csv(args.output_dir / "orthogonality_report.csv", index=False)
    cv.to_csv(args.output_dir / "cross_validation.csv", index=False)
    coefficients.to_csv(args.output_dir / "fitted_coefficients.csv", index=False)
    evaluation_metrics.to_csv(args.output_dir / "evaluation_metrics.csv", index=False)
    alpha_metrics.to_csv(args.output_dir / "alpha_evaluation.csv", index=False)
    ranking_significance.to_csv(args.output_dir / "ranking_significance.csv", index=False)

    save_figures(lambda_values, figure_dir)
    figure_files = [
        "figures/lambda_distribution.svg",
        "figures/lambda_vs_contribution.svg",
        "figures/lambda_vs_quality.svg",
        "figures/form_a_vs_form_b.svg",
    ]
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
        selected_alpha=selected_alpha,
        lambda_calibrations=lambda_calibrations,
        evaluation_config={
            "permutations": args.ranking_permutations,
            "permutation_seed": args.ranking_permutation_seed,
        },
        figure_files=figure_files,
    )

    manifest = {
        "project": "Project 2",
        "experiment": "Experiment 2",
        "source_experiment": str(args.exp1_dir),
        "output_dir": str(args.output_dir),
        "dataset_manifest_file": "dataset_manifest.json",
        "dataset_manifest": dataset_manifest,
        "source_dataset_provenance": exp1_manifest.get("dataset_provenance", {}),
        "is_synthetic_present": bool(df["is_synthetic"].astype(bool).any()),
        "environment": environment_manifest(),
        "form_a_features": FORM_A_FEATURES,
        "form_b_features": FORM_B_FEATURES,
        "ridge_alpha_grid": ridge_alphas,
        "extended_ridge_alpha_candidates": EXTENDED_RIDGE_ALPHAS,
        "selected_ridge_alpha": selected_alpha,
        "selected_ridge_alpha_on_boundary": selected_alpha in {
            min(ridge_alphas),
            max(ridge_alphas),
        },
        "alpha_selection_rule": "minimum mean leave-one-task-out RMSE",
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
        "outputs": [
            "lambda_values.csv",
            "lambda_validation.csv",
            "orthogonality_report.csv",
            "cross_validation.csv",
            "fitted_coefficients.csv",
            "evaluation_metrics.csv",
            "alpha_evaluation.csv",
            "ranking_significance.csv",
            "comparison_report.md",
            "figures/",
        ],
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
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("=== Experiment 2 Complete ===")
    print(f"Rows used: {len(df)}")
    print(f"Selected ridge alpha: {selected_alpha}")
    print(f"Lambda calibration: {lambda_calibrations}")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
