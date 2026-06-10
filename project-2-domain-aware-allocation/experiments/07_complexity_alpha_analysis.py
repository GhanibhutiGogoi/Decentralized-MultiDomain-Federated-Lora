"""
Experiment 08: Complexity Components vs Alpha Analysis

Purpose:
    Analyze which dataset complexity components are related to the oracle
    LoRA alpha values obtained from Experiment 05.

Inputs:
    - results/experiment_01_data_and_complexity/complexity_scores.json
    - results/experiment_05_alpha_search/best_alpha_per_client.json

Outputs:
    - results/experiment_08_complexity_alpha_analysis/complexity_alpha_analysis.json
    - results/experiment_08_complexity_alpha_analysis/complexity_alpha_scatter.png
"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau


COMPLEXITY_PATH = "results/experiment_01_data_and_complexity/complexity_scores.json"
ALPHA_PATH = "results/experiment_05_alpha_search/best_alpha_per_client.json"

OUTPUT_DIR = "results/experiment_08_complexity_alpha_analysis"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "complexity_alpha_analysis.json")
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "complexity_alpha_scatter.png")


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r") as f:
        return json.load(f)


def compute_corr(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    return {
        "pearson": {
            "r": float(pearsonr(x, y).statistic),
            "p_value": float(pearsonr(x, y).pvalue),
        },
        "spearman": {
            "rho": float(spearmanr(x, y).statistic),
            "p_value": float(spearmanr(x, y).pvalue),
        },
        "kendall": {
            "tau": float(kendalltau(x, y).statistic),
            "p_value": float(kendalltau(x, y).pvalue),
        },
    }


def build_client_rows(complexity_data, alpha_data):
    rows = []

    for client_id in sorted(complexity_data.keys(), key=lambda x: int(x)):
        complexity = complexity_data[client_id]
        alpha = alpha_data[client_id]

        rows.append({
            "client_id": int(client_id),
            "domain_id": complexity["domain_id"],
            "domain_name": complexity["domain_name"],

            "complexity_score": complexity["score"],
            "entropy": complexity["entropy"],
            "diversity": complexity["diversity"],
            "intrinsic_dim": complexity["intrinsic_dim"],
            "task_difficulty": complexity["task_difficulty"],
            "data_imbalance": complexity["data_imbalance"],
            "n_samples": complexity["n_samples"],
            "n_classes": complexity["n_classes"],

            "fixed_rank": alpha["fixed_rank"],
            "best_alpha": int(alpha["best_alpha"]),
            "best_accuracy": alpha["best_accuracy"],
        })

    return rows


def analyze_correlations(rows):
    alpha_values = [row["best_alpha"] for row in rows]

    metrics = {
        "complexity_score": [row["complexity_score"] for row in rows],
        "entropy": [row["entropy"] for row in rows],
        "diversity": [row["diversity"] for row in rows],
        "intrinsic_dim": [row["intrinsic_dim"] for row in rows],
        "task_difficulty": [row["task_difficulty"] for row in rows],
        "data_imbalance": [row["data_imbalance"] for row in rows],
        "n_samples": [row["n_samples"] for row in rows],
        "n_classes": [row["n_classes"] for row in rows],
    }

    correlations = {}

    for metric_name, values in metrics.items():
        if len(set(values)) <= 1:
            correlations[metric_name] = {
                "note": "constant metric, correlation is undefined"
            }
        else:
            correlations[metric_name] = compute_corr(values, alpha_values)

    return correlations


def summarize(correlations):
    ranked_by_abs_pearson = []

    for metric, result in correlations.items():
        if "pearson" not in result:
            continue

        ranked_by_abs_pearson.append({
            "metric": metric,
            "pearson_r": result["pearson"]["r"],
            "pearson_p_value": result["pearson"]["p_value"],
            "abs_pearson_r": abs(result["pearson"]["r"]),
        })

    ranked_by_abs_pearson = sorted(
        ranked_by_abs_pearson,
        key=lambda x: x["abs_pearson_r"],
        reverse=True,
    )

    return {
        "strongest_metrics_by_abs_pearson": ranked_by_abs_pearson,
    }


def plot_complexity_vs_alpha(rows):
    x = [row["complexity_score"] for row in rows]
    y = [row["best_alpha"] for row in rows]
    client_ids = [row["client_id"] for row in rows]

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y)

    for xi, yi, cid in zip(x, y, client_ids):
        plt.text(xi, yi, str(cid), fontsize=8, ha="left", va="bottom")

    plt.xlabel("Complexity Score")
    plt.ylabel("Best Alpha")
    plt.title("Complexity Score vs Best Alpha")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    complexity_data = load_json(COMPLEXITY_PATH)
    alpha_data = load_json(ALPHA_PATH)

    rows = build_client_rows(
        complexity_data=complexity_data,
        alpha_data=alpha_data,
    )

    correlations = analyze_correlations(rows)
    summary = summarize(correlations)

    report = {
        "description": (
            "Correlation analysis between dataset complexity components "
            "and oracle LoRA alpha values."
        ),
        "client_table": rows,
        "correlations_with_best_alpha": correlations,
        "summary": summary,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    plot_complexity_vs_alpha(rows)

    print("=" * 60)
    print("EXPERIMENT 08 COMPLETE")
    print("=" * 60)

    print("\nStrongest metrics by absolute Pearson correlation:")
    for item in summary["strongest_metrics_by_abs_pearson"]:
        print(
            f"{item['metric']}: "
            f"r={item['pearson_r']:.4f}, "
            f"p={item['pearson_p_value']:.4f}"
        )

    print(f"\nSaved JSON: {OUTPUT_JSON}")
    print(f"Saved Plot: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
