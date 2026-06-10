"""
Experiment 07: LoRA Capacity Analysis

Purpose:
    Combine oracle rank and oracle alpha into a single capacity measure:

        capacity = rank * alpha

    Then analyze whether client complexity predicts LoRA capacity better
    than it predicted rank alone.

Inputs:
    - results/experiment_01_data_and_complexity/complexity_scores.json
    - results/experiment_02_oracle_rank_search/oracle_results.json
    - results/experiment_05_alpha_search/best_alpha_per_client.json

Outputs:
    - results/experiment_07_capacity_analysis/capacity_analysis.json
    - results/experiment_07_capacity_analysis/complexity_capacity_scatter.png
"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau


COMPLEXITY_PATH = "results/experiment_01_data_and_complexity/complexity_scores.json"
ORACLE_RANK_PATH = "results/experiment_02_oracle_rank_search/oracle_results.json"
ALPHA_PATH = "results/experiment_05_alpha_search/best_alpha_per_client.json"

OUTPUT_DIR = "results/experiment_07_capacity_analysis"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "capacity_analysis.json")
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "complexity_capacity_scatter.png")


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r") as f:
        return json.load(f)


def get_best_rank(rank_results):
    return int(max(rank_results, key=rank_results.get))


def safe_corr(x, y):
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


def plot_complexity_capacity(rows):
    complexities = [row["complexity_score"] for row in rows]
    capacities = [row["capacity"] for row in rows]
    client_ids = [row["client_id"] for row in rows]

    plt.figure(figsize=(8, 6))
    plt.scatter(complexities, capacities)

    for x, y, cid in zip(complexities, capacities, client_ids):
        plt.text(x, y, str(cid), fontsize=8, ha="right", va="bottom")

    plt.xlabel("Complexity Score")
    plt.ylabel("LoRA Capacity (Rank × Alpha)")
    plt.title("Complexity Score vs LoRA Capacity")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    complexity_data = load_json(COMPLEXITY_PATH)
    oracle_rank_data = load_json(ORACLE_RANK_PATH)
    alpha_data = load_json(ALPHA_PATH)

    rows = []

    for client_id in sorted(complexity_data.keys(), key=lambda x: int(x)):
        complexity_score = complexity_data[client_id]["score"]
        best_rank = get_best_rank(oracle_rank_data[client_id])
        best_alpha = int(alpha_data[client_id]["best_alpha"])

        capacity = best_rank * best_alpha

        rows.append({
            "client_id": int(client_id),
            "complexity_score": complexity_score,
            "best_rank": best_rank,
            "best_alpha": best_alpha,
            "capacity": capacity,
            "domain_id": complexity_data[client_id]["domain_id"],
            "domain_name": complexity_data[client_id]["domain_name"],
        })

    complexities = [row["complexity_score"] for row in rows]
    ranks = [row["best_rank"] for row in rows]
    alphas = [row["best_alpha"] for row in rows]
    capacities = [row["capacity"] for row in rows]

    report = {
        "description": "LoRA capacity is defined as best_rank * best_alpha.",
        "client_capacity_table": rows,
        "correlations": {
            "complexity_vs_rank": safe_corr(complexities, ranks),
            "complexity_vs_alpha": safe_corr(complexities, alphas),
            "complexity_vs_capacity": safe_corr(complexities, capacities),
            "rank_vs_alpha": safe_corr(ranks, alphas),
        },
        "summary": {
            "min_capacity": min(capacities),
            "max_capacity": max(capacities),
            "mean_capacity": float(np.mean(capacities)),
            "unique_capacities": sorted(list(set(capacities))),
        },
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    plot_complexity_capacity(rows)

    print("=" * 60)
    print("EXPERIMENT 07 COMPLETE")
    print("=" * 60)

    print("\nComplexity vs Capacity:")
    print(report["correlations"]["complexity_vs_capacity"])

    print("\nRank vs Alpha:")
    print(report["correlations"]["rank_vs_alpha"])

    print("\nSummary:")
    print(json.dumps(report["summary"], indent=2))

    print(f"\nSaved JSON: {OUTPUT_JSON}")
    print(f"Saved Plot: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
