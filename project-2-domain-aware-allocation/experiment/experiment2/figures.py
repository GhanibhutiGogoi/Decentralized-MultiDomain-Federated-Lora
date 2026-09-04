"""Automatic SVG figures for Experiment 2 evaluation outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FORM_COLORS = {"form_a": "#1f77b4", "form_b": "#d62728"}


def _scale(values: np.ndarray, lower: float, upper: float, span: float) -> np.ndarray:
    if values.size == 0:
        return values
    if abs(upper - lower) < 1e-12:
        upper = lower + 1.0
    return (values - lower) / (upper - lower) * span


def _svg_bar_chart(
    path: Path,
    df: pd.DataFrame,
    *,
    value_col: str,
    title: str,
    y_label: str,
) -> None:
    width, height = 760, 460
    margin_left, margin_bottom, margin_top = 72, 86, 50
    plot_w = width - margin_left - 32
    plot_h = height - margin_top - margin_bottom
    forms = list(df["form"])
    values = df[value_col].to_numpy(dtype=float)
    y_min = min(0.0, float(np.nanmin(values)))
    y_max = max(0.0, float(np.nanmax(values)))
    bar_w = min(76, plot_w / max(len(values), 1) * 0.6)
    step = plot_w / max(len(values), 1)

    bars = []
    for idx, row in df.reset_index(drop=True).iterrows():
        value = float(row[value_col])
        scaled = float(_scale(np.asarray([value]), y_min, y_max, plot_h)[0])
        baseline = margin_top + plot_h - float(_scale(np.asarray([0.0]), y_min, y_max, plot_h)[0])
        y = margin_top + plot_h - scaled
        height_value = abs(baseline - y)
        x = margin_left + idx * step + (step - bar_w) / 2
        label = str(row["form"]).replace("_", " ").title()
        color = FORM_COLORS.get(str(row["form"]), "#555555")
        bars.append(
            f'<rect x="{x:.2f}" y="{min(y, baseline):.2f}" width="{bar_w:.2f}" '
            f'height="{height_value:.2f}" fill="{color}" fill-opacity="0.78"/>'
        )
        bars.append(
            f'<text x="{x + bar_w / 2:.2f}" y="{height - 48}" '
            f'text-anchor="middle" font-size="12" font-family="Arial">{label}</text>'
        )
        bars.append(
            f'<text x="{x + bar_w / 2:.2f}" y="{min(y, baseline) - 6:.2f}" '
            f'text-anchor="middle" font-size="11" font-family="Arial">{value:.3g}</text>'
        )

    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="28" text-anchor="middle" font-size="20" font-family="Arial">{title}</text>
<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - 32}" y2="{margin_top + plot_h}" stroke="#222"/>
<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#222"/>
<text x="20" y="{margin_top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 20 {margin_top + plot_h / 2})" font-size="13" font-family="Arial">{y_label}</text>
<text x="{margin_left - 8}" y="{margin_top + plot_h}" text-anchor="end" font-size="11" font-family="Arial">{y_min:.3g}</text>
<text x="{margin_left - 8}" y="{margin_top}" text-anchor="end" font-size="11" font-family="Arial">{y_max:.3g}</text>
{chr(10).join(bars)}
</svg>
""",
        encoding="utf-8",
    )


def _global_metrics(evaluation_metrics: pd.DataFrame) -> pd.DataFrame:
    return evaluation_metrics[
        (evaluation_metrics["scope"] == "all")
        & (evaluation_metrics["scope_value"] == "ALL")
    ].copy()


def _svg_alpha_comparison(path: Path, alpha_metrics: pd.DataFrame) -> None:
    form_b = alpha_metrics[alpha_metrics["form"] == "form_b"].copy()
    if form_b.empty:
        return
    width, height = 820, 460
    margin = 64
    x_values = form_b["ridge_alpha"].astype(float).to_numpy()
    y_values = form_b["rmse"].to_numpy(dtype=float)
    x_log = np.log10(x_values)
    x_scaled = margin + _scale(x_log, float(x_log.min()), float(x_log.max()), width - 2 * margin)
    y_scaled = height - margin - _scale(
        y_values,
        float(y_values.min()),
        float(y_values.max()),
        height - 2 * margin,
    )
    points = [
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#d62728" fill-opacity="0.8"/>'
        for x, y in zip(x_scaled, y_scaled)
    ]
    labels = [
        f'<text x="{x:.2f}" y="{height - 36}" text-anchor="middle" font-size="11" font-family="Arial">{alpha:g}</text>'
        for x, alpha in zip(x_scaled, x_values)
    ]
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="28" text-anchor="middle" font-size="20" font-family="Arial">Ridge Alpha Comparison</text>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#222"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#222"/>
<polyline points="{' '.join(f'{x:.2f},{y:.2f}' for x, y in zip(x_scaled, y_scaled))}" fill="none" stroke="#d62728" stroke-width="2"/>
{chr(10).join(points)}
{chr(10).join(labels)}
<text x="{width / 2}" y="{height - 12}" text-anchor="middle" font-size="13" font-family="Arial">ridge_alpha</text>
<text x="18" y="{height / 2}" text-anchor="middle" transform="rotate(-90 18 {height / 2})" font-size="13" font-family="Arial">mean RMSE</text>
</svg>
""",
        encoding="utf-8",
    )


def save_evaluation_figures(
    figure_dir: Path,
    *,
    evaluation_metrics: pd.DataFrame,
    alpha_metrics: pd.DataFrame,
) -> list[str]:
    """Write automatic evaluation figures and return relative file names."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    global_metrics = _global_metrics(evaluation_metrics)
    if not global_metrics.empty:
        _svg_bar_chart(
            figure_dir / "regression_performance.svg",
            global_metrics,
            value_col="rmse",
            title="Regression Performance",
            y_label="RMSE",
        )
        written.append("figures/regression_performance.svg")
        _svg_bar_chart(
            figure_dir / "ranking_performance.svg",
            global_metrics,
            value_col="pairwise_ranking_accuracy",
            title="Ranking Performance",
            y_label="Pairwise ranking accuracy",
        )
        written.append("figures/ranking_performance.svg")
        _svg_bar_chart(
            figure_dir / "form_comparison.svg",
            global_metrics,
            value_col="spearman",
            title="Form A vs Form B Ranking",
            y_label="Spearman",
        )
        written.append("figures/form_comparison.svg")

    _svg_alpha_comparison(figure_dir / "alpha_comparison.svg", alpha_metrics)
    if (figure_dir / "alpha_comparison.svg").exists():
        written.append("figures/alpha_comparison.svg")
    return written
