"""
Experiment 03: Correlation Analysis — Complexity Score vs. Oracle Rank

Purpose:
    Test the fundamental hypothesis: does the domain complexity score
    predict the optimal LoRA rank?

    Loads results from experiments 01 and 02, computes statistical
    correlations, and produces the key scatter plot for the paper.

Outputs:
    - results/complexity_vs_rank.png (KEY FIGURE)
    - results/submetic_correlations.png
    - results/correlation_analysis.json
    - Console report with R², Spearman, significance

Runtime: seconds (just loading and statistics)

Success Criteria:
    - Spearman ρ > 0.6 with p < 0.05
"""

import os
import sys
import json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.visualization import plot_complexity_vs_oracle_rank

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results'
    )

    # Load complexity scores from experiment 01
    complexity_path = os.path.join(results_dir, 'complexity_scores.json')
    if not os.path.exists(complexity_path):
        print("ERROR: complexity_scores.json not found. Run experiment 01 first.")
        return

    with open(complexity_path) as f:
        complexity_data = json.load(f)

    # Load oracle results from experiment 02
    oracle_path = os.path.join(results_dir, 'oracle_results.json')
    if not os.path.exists(oracle_path):
        print("ERROR: oracle_results.json not found. Run experiment 02 first.")
        return

    with open(oracle_path) as f:
        oracle_data = json.load(f)

    # Extract data
    client_ids = sorted(complexity_data.keys(), key=int)

    complexity_scores = {}
    oracle_ranks = {}
    domain_ids = {}

    for cid in client_ids:
        complexity_scores[int(cid)] = complexity_data[cid]['score']
        domain_ids[int(cid)] = complexity_data[cid]['domain_id']

        # Find best rank for this client
        rank_accs = oracle_data[cid]
        best_rank = max(rank_accs, key=rank_accs.get)
        oracle_ranks[int(cid)] = int(best_rank)

    # === Main Correlation Analysis ===
    x = [complexity_scores[int(cid)] for cid in client_ids]
    y = [oracle_ranks[int(cid)] for cid in client_ids]

    # Pearson
    pearson_r, pearson_p = stats.pearsonr(x, y)
    r_squared = pearson_r ** 2

    # Spearman
    spearman_rho, spearman_p = stats.spearmanr(x, y)

    # Kendall's tau
    kendall_tau, kendall_p = stats.kendalltau(x, y)

    # Linear regression
    slope, intercept, _, _, std_err = stats.linregress(x, y)

    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS: Complexity Score vs. Oracle Rank")
    print("=" * 60)
    print(f"\n  Pearson r:     {pearson_r:.4f}  (p = {pearson_p:.6f})")
    print(f"  R-squared:     {r_squared:.4f}")
    print(f"  Spearman ρ:    {spearman_rho:.4f}  (p = {spearman_p:.6f})")
    print(f"  Kendall τ:     {kendall_tau:.4f}  (p = {kendall_p:.6f})")
    print(f"\n  Linear fit:    rank = {slope:.2f} * complexity + {intercept:.2f}")
    print(f"  Slope SE:      {std_err:.2f}")

    # Verdict
    print("\n  " + "-" * 40)
    if spearman_rho > 0.6 and spearman_p < 0.05:
        print("  RESULT: HYPOTHESIS SUPPORTED")
        print("  Complexity score significantly predicts optimal rank.")
    elif spearman_rho > 0.4:
        print("  RESULT: MODERATE CORRELATION")
        print("  Some predictive power but consider refining sub-metric weights.")
    else:
        print("  RESULT: WEAK CORRELATION")
        print("  Complexity metric needs improvement. Check sub-metric analysis below.")
    print("  " + "-" * 40)

    # === Sub-metric Analysis ===
    print("\n\nPer sub-metric correlation with oracle rank:")
    print(f"  {'Metric':>20} | {'Spearman ρ':>12} | {'p-value':>10} | {'Significant':>11}")
    print("  " + "-" * 60)

    submetrics = ['entropy', 'diversity', 'intrinsic_dim', 'task_difficulty', 'data_imbalance']
    submetric_results = {}

    for metric in submetrics:
        values = [complexity_data[cid][metric] for cid in client_ids]
        rho, p = stats.spearmanr(values, y)
        sig = "Yes" if p < 0.05 else "No"
        print(f"  {metric:>20} | {rho:12.4f} | {p:10.6f} | {sig:>11}")
        submetric_results[metric] = {'spearman_rho': rho, 'p_value': p}

    # === Plot sub-metric correlations ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    metric_labels = ['Label Entropy', 'Feature Diversity', 'Intrinsic Dim',
                     'Task Difficulty', 'Data Imbalance']

    for i, (metric, label) in enumerate(zip(submetrics, metric_labels)):
        ax = axes[i]
        values = [complexity_data[cid][metric] for cid in client_ids]
        domains = [complexity_data[cid]['domain_id'] for cid in client_ids]

        scatter = ax.scatter(values, y, c=domains, cmap='tab10', s=80,
                           edgecolors='black', linewidth=0.5)

        # Regression line
        slope_m, intercept_m, _, _, _ = stats.linregress(values, y)
        x_line = np.linspace(min(values), max(values), 50)
        ax.plot(x_line, slope_m * x_line + intercept_m, 'r--', alpha=0.7)

        rho = submetric_results[metric]['spearman_rho']
        p = submetric_results[metric]['p_value']
        ax.set_title(f'{label}\nρ={rho:.3f}, p={p:.4f}', fontsize=11)
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel('Oracle Rank', fontsize=10)
        ax.set_yticks([4, 8, 16, 32, 64])
        ax.grid(True, alpha=0.3)

    # Composite score in last panel
    ax = axes[5]
    ax.scatter(x, y, c=[domain_ids[int(cid)] for cid in client_ids],
              cmap='tab10', s=80, edgecolors='black', linewidth=0.5)
    slope_c, intercept_c, _, _, _ = stats.linregress(x, y)
    x_line = np.linspace(min(x), max(x), 50)
    ax.plot(x_line, slope_c * x_line + intercept_c, 'r--', alpha=0.7)
    ax.set_title(f'Composite Score\nρ={spearman_rho:.3f}, p={spearman_p:.4f}', fontsize=11)
    ax.set_xlabel('Composite Score', fontsize=10)
    ax.set_ylabel('Oracle Rank', fontsize=10)
    ax.set_yticks([4, 8, 16, 32, 64])
    ax.grid(True, alpha=0.3)

    plt.suptitle('Sub-metric Correlations with Oracle Optimal Rank', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'submetric_correlations.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # === Main scatter plot (the key figure) ===
    plot_complexity_vs_oracle_rank(
        complexity_scores,
        oracle_ranks,
        domain_ids=domain_ids,
        save_path=os.path.join(results_dir, 'complexity_vs_rank.png'),
    )

    # === Save analysis results ===
    analysis = {
        'pearson_r': pearson_r,
        'r_squared': r_squared,
        'spearman_rho': spearman_rho,
        'spearman_p': spearman_p,
        'kendall_tau': kendall_tau,
        'kendall_p': kendall_p,
        'linear_slope': slope,
        'linear_intercept': intercept,
        'per_submetric': submetric_results,
        'complexity_scores': {str(k): v for k, v in complexity_scores.items()},
        'oracle_ranks': {str(k): v for k, v in oracle_ranks.items()},
    }

    output_path = os.path.join(results_dir, 'correlation_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nCorrelation analysis saved to {output_path}")

    print("\nExperiment 03 complete!")


if __name__ == '__main__':
    main()
