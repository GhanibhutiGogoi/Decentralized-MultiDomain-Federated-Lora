"""
Experiment 08: Scaling Factor Analysis

Purpose:
    LoRA uses an effective scaling factor:

        scaling = alpha / rank

    Previous experiments searched for oracle rank and oracle alpha separately.
    This experiment analyzes whether the effective LoRA scaling factor is related
    to rank, alpha, and dataset complexity characteristics.

Inputs:
    - results/experiment_01_data_and_complexity/complexity_scores.json
    - results/experiment_02_oracle_rank_search/oracle_results.json
    - results/experiment_05_alpha_search/best_alpha_per_client.json

Outputs:
    - results/experiment_08_scaling_analysis/scaling_analysis.json
    - results/experiment_08_scaling_analysis/complexity_scaling_scatter.png
"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau


COMPLEXITY_PATH = "results/experiment_01_data_and_complexity/complexity_scores.json"
ORACLE_RANK_PATH = "results/experiment_02_oracle_rank_search/oracle_results.json"
ALPHA_PATH = "results/experiment_05_alpha_search/best_alpha_per_client.json"

OUTPUT_DIR = "results/experiment_08_scaling_analysis"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "scaling_analysis.json")
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "complexity_scaling_scatter.png")


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r") as f:
        return json.load(f)


def get_best_rank(rank_results):
    return int(max(rank_results, key=rank_results.get))


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


def build_client_rows(complexity_data, oracle_rank_data, alpha_data):
    rows = []

    for client_id in sorted(complexity_data.keys(), key=lambda x: int(x)):
        best_rank = get_best_rank(oracle_rank_data[client_id])
        best_alpha = int(alpha_data[client_id]["best_alpha"])
        scaling = best_alpha / best_rank

        rows.append({
            "client_id": int(client_id),
            "domain_id": complexity_data[client_id]["domain_id"],
            "domain_name": complexity_data[client_id]["domain_name"],

            "complexity_score": complexity_data[client_id]["score"],
            "entropy": complexity_data[client_id]["entropy"],
            "diversity": complexity_data[client_id]["diversity"],
            "intrinsic_dim": complexity_data[client_id]["intrinsic_dim"],
            "data_imbalance": complexity_data[client_id]["data_imbalance"],

            "best_rank": best_rank,
            "best_alpha": best_alpha,
            "scaling": scaling,
        })

    return rows


def analyze_correlations(rows):
    complexity_scores = [row["complexity_score"] for row in rows]
    entropy_values = [row["entropy"] for row in rows]
    diversity_values = [row["diversity"] for row in rows]
    intrinsic_dim_values = [row["intrinsic_dim"] for row in rows]
    imbalance_values = [row["data_imbalance"] for row in rows]

    ranks = [row["best_rank"] for row in rows]
    alphas = [row["best_alpha"] for row in rows]
    scalings = [row["scaling"] for row in rows]

    return {
        "complexity_vs_scaling": compute_corr(complexity_scores, scalings),
        "entropy_vs_scaling": compute_corr(entropy_values, scalings),
        "diversity_vs_scaling": compute_corr(diversity_values, scalings),
        "intrinsic_dim_vs_scaling": compute_corr(intrinsic_dim_values, scalings),
        "data_imbalance_vs_scaling": compute_corr(imbalance_values, scalings),
        "rank_vs_best_alpha": compute_corr(best_ranks, best_alphas),
        "alpha_vs_scaling": compute_corr(alphas, scalings),
        "rank_vs_alpha": compute_corr(ranks, alphas),
    }


def summarize(rows, correlations):
    scalings = [row["scaling"] for row in rows]

    ranked_correlations = []

    for name, result in correlations.items():
        if "pearson" not in result:
            continue

        ranked_correlations.append({
            "relationship": name,
            "pearson_r": result["pearson"]["r"],
            "pearson_p_value": result["pearson"]["p_value"],
            "spearman_rho": result["spearman"]["rho"],
            "spearman_p_value": result["spearman"]["p_value"],
            "kendall_tau": result["kendall"]["tau"],
            "kendall_p_value": result["kendall"]["p_value"],
            "abs_spearman": abs(result["spearman"]["rho"]),
        })

    ranked_correlations = sorted(
        ranked_correlations,
        key=lambda x: x["abs_spearman"],
        reverse=True,
    )

    return {
        "unique_scaling_values": sorted(list(set(scalings))),
        "mean_scaling": float(np.mean(scalings)),
        "min_scaling": float(np.min(scalings)),
        "max_scaling": float(np.max(scalings)),
        "strongest_relationships_by_abs_spearman": ranked_correlations,
    }


def plot_complexity_vs_scaling(rows):
    x = [row["complexity_score"] for row in rows]
    y = [row["scaling"] for row in rows]
    client_ids = [row["client_id"] for row in rows]

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y)

    for xi, yi, cid in zip(x, y, client_ids):
        plt.text(xi, yi, str(cid), fontsize=8, ha="left", va="bottom")

    plt.xlabel("Complexity Score")
    plt.ylabel("LoRA Scaling (Alpha / Rank)")
    plt.title("Complexity Score vs LoRA Scaling")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    complexity_data = load_json(COMPLEXITY_PATH)
    oracle_rank_data = load_json(ORACLE_RANK_PATH)
    alpha_data = load_json(ALPHA_PATH)

    rows = build_client_rows(
        complexity_data=complexity_data,
        oracle_rank_data=oracle_rank_data,
        alpha_data=alpha_data,
    )

    correlations = analyze_correlations(rows)
    summary = summarize(rows, correlations)

    report = {
        "description": "LoRA scaling is defined as alpha / rank.",
        "client_table": rows,
        "correlations": correlations,
        "summary": summary,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    plot_complexity_vs_scaling(rows)

    print("=" * 60)
    print("EXPERIMENT 08 COMPLETE")
    print("=" * 60)

    print("\nScaling summary:")
    print(json.dumps({
        "unique_scaling_values": summary["unique_scaling_values"],
        "mean_scaling": summary["mean_scaling"],
        "min_scaling": summary["min_scaling"],
        "max_scaling": summary["max_scaling"],
    }, indent=2))

    print("\nStrongest relationships by absolute Spearman correlation:")
    for item in summary["strongest_relationships_by_abs_spearman"]:
        print(
            f"{item['relationship']}: "
            f"Spearman={item['spearman_rho']:.4f}, "
            f"p={item['spearman_p_value']:.4f}"
        )

    print(f"\nSaved JSON: {OUTPUT_JSON}")
    print(f"Saved Plot: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
