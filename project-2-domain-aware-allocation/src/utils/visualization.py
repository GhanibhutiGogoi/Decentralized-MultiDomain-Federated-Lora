"""
Visualization utilities for domain-aware LoRA allocation experiments.

Includes standard FL plots (convergence, per-domain accuracy) plus
Project 2-specific plots (complexity analysis, oracle heatmap, allocation comparison).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_curves(histories, labels, save_path=None):
    """
    Plot accuracy curves for multiple experiments.

    Args:
        histories: list of history dicts (each with 'rounds' and 'avg_accuracy')
        labels: list of names for each experiment
        save_path: optional path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for history, label in zip(histories, labels):
        ax.plot(history['rounds'], history['avg_accuracy'], label=label, linewidth=2)

    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Average Accuracy', fontsize=12)
    ax.set_title('Federated Learning Convergence', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.close()


def plot_per_domain_accuracy(history, domain_names=None, save_path=None):
    """Plot per-domain accuracy over rounds."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    rounds = history['rounds']
    domain_ids = sorted(history['per_domain_accuracy'][0].keys())

    for did in domain_ids:
        accs = [r[did] for r in history['per_domain_accuracy']]
        label = domain_names[did] if domain_names else f"Domain {did}"
        ax.plot(rounds, accs, label=label, linewidth=2)

    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Per-Domain Accuracy', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_data_distribution(clients_data, save_path=None):
    """Plot data distribution across clients showing non-IID-ness."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    client_ids = sorted(clients_data.keys())
    n_samples = [clients_data[cid]['n_samples'] for cid in client_ids]
    domains = [clients_data[cid]['domain_id'] for cid in client_ids]

    colors = plt.cm.tab10(np.array(domains) / max(max(domains), 1))
    ax.bar(range(len(client_ids)), n_samples, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Client ID', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Data Distribution Across Clients', fontsize=14)
    ax.set_xticks(range(len(client_ids)))
    ax.set_xticklabels([str(cid) for cid in client_ids])

    unique_domains = sorted(set(domains))
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=plt.cm.tab10(d / max(max(domains), 1)))
        for d in unique_domains
    ]
    legend_labels = [
        clients_data[next(cid for cid in client_ids if clients_data[cid]['domain_id'] == d)]['domain_name']
        for d in unique_domains
    ]
    ax.legend(legend_handles, legend_labels, title="Domain", loc='upper right')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# Project 2 - Specific Visualizations
# ============================================================

def plot_complexity_breakdown(client_complexities, save_path=None):
    """
    Stacked bar chart showing the 5 sub-metrics for each client,
    grouped by domain. Shows which factors drive complexity.

    Args:
        client_complexities: dict mapping client_id -> complexity result dict
        save_path: optional path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    client_ids = sorted(client_complexities.keys())
    metrics = ['entropy', 'diversity', 'intrinsic_dim', 'task_difficulty', 'data_imbalance']
    metric_labels = ['Label Entropy', 'Feature Diversity', 'Intrinsic Dim', 'Task Difficulty', 'Data Imbalance']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']

    # Default weights for showing weighted contribution
    weights = [0.3, 0.2, 0.2, 0.2, 0.1]

    x = np.arange(len(client_ids))
    bottom = np.zeros(len(client_ids))

    for metric, label, color, w in zip(metrics, metric_labels, colors, weights):
        values = [client_complexities[cid][metric] * w for cid in client_ids]
        ax.bar(x, values, bottom=bottom, label=f"{label} (w={w})",
               color=color, edgecolor='white', linewidth=0.5)
        bottom += values

    # Add total score line
    scores = [client_complexities[cid]['score'] for cid in client_ids]
    ax.plot(x, scores, 'k--o', label='Composite Score', linewidth=2, markersize=6)

    # Domain annotations
    domains = [client_complexities[cid]['domain_id'] for cid in client_ids]
    for i, (cid, d) in enumerate(zip(client_ids, domains)):
        ax.text(i, -0.03, f"D{d}", ha='center', va='top', fontsize=8, color='gray')

    ax.set_xlabel('Client ID', fontsize=12)
    ax.set_ylabel('Weighted Complexity Contribution', fontsize=12)
    ax.set_title('Domain Complexity Breakdown by Client', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(cid) for cid in client_ids])
    ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.01, 1))
    ax.set_ylim(-0.05, max(scores) * 1.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.close()


def plot_complexity_vs_oracle_rank(complexity_scores, oracle_ranks, domain_ids=None, save_path=None):
    """
    Scatter plot: complexity score (x) vs oracle-optimal rank (y).

    This is the KEY FIGURE for the paper: shows that complexity predicts rank.

    Args:
        complexity_scores: dict mapping client_id -> float
        oracle_ranks: dict mapping client_id -> int
        domain_ids: optional dict mapping client_id -> domain_id (for coloring)
        save_path: optional path to save figure
    """
    from scipy import stats

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    client_ids = sorted(complexity_scores.keys())
    x = [complexity_scores[cid] for cid in client_ids]
    y = [oracle_ranks[cid] for cid in client_ids]

    # Color by domain if available
    if domain_ids:
        c = [domain_ids[cid] for cid in client_ids]
        scatter = ax.scatter(x, y, c=c, cmap='tab10', s=120, edgecolors='black',
                           linewidth=1, zorder=3)
        plt.colorbar(scatter, ax=ax, label='Domain ID')
    else:
        ax.scatter(x, y, s=120, edgecolors='black', linewidth=1, zorder=3, color='steelblue')

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_line = np.linspace(min(x) - 0.05, max(x) + 0.05, 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.7, label='Linear fit')

    # Spearman correlation
    spearman_rho, spearman_p = stats.spearmanr(x, y)

    # Annotate
    textstr = (f'R² = {r_value**2:.3f}\n'
               f'Spearman ρ = {spearman_rho:.3f}\n'
               f'p-value = {spearman_p:.4f}')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    # Client labels
    for cid, xi, yi in zip(client_ids, x, y):
        ax.annotate(str(cid), (xi, yi), textcoords="offset points",
                   xytext=(5, 5), fontsize=8)

    ax.set_xlabel('Domain Complexity Score', fontsize=13)
    ax.set_ylabel('Oracle Optimal Rank', fontsize=13)
    ax.set_title('Complexity Score vs. Optimal LoRA Rank', fontsize=14)
    ax.set_yticks([4, 8, 16, 32, 64])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.close()


def plot_oracle_heatmap(oracle_results, save_path=None):
    """
    Heatmap: clients (y) x ranks (x), cell color = accuracy.

    Shows the rank-accuracy landscape for each client. The optimal rank
    per client is highlighted.

    Args:
        oracle_results: dict mapping client_id -> {rank -> accuracy}
        save_path: optional path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    client_ids = sorted(oracle_results.keys(), key=int)
    ranks = sorted(set(int(r) for cid in client_ids for r in oracle_results[cid].keys()))

    # Build matrix
    matrix = []
    y_labels = []
    for cid in client_ids:
        row = [oracle_results[cid].get(str(r), oracle_results[cid].get(r, 0))
               for r in ranks]
        matrix.append(row)
        best_rank = max(oracle_results[cid], key=oracle_results[cid].get)
        y_labels.append(f"Client {cid} (best={best_rank})")

    matrix = np.array(matrix)

    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=[str(r) for r in ranks],
                yticklabels=y_labels, ax=ax,
                linewidths=0.5, cbar_kws={'label': 'Test Accuracy'})

    ax.set_xlabel('LoRA Rank', fontsize=12)
    ax.set_ylabel('Client', fontsize=12)
    ax.set_title('Oracle Rank Search: Accuracy per Client x Rank', fontsize=14)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.close()


def plot_allocation_comparison(results_dict, save_path=None):
    """
    Grouped bar chart comparing final accuracy across allocation strategies.

    Args:
        results_dict: dict mapping strategy_name -> {
            'avg_accuracy': float,
            'per_domain': dict mapping domain_id -> accuracy,
        }
        save_path: optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    strategies = list(results_dict.keys())
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']

    # Panel 1: Average accuracy
    avg_accs = [results_dict[s]['avg_accuracy'] for s in strategies]
    bars = axes[0].bar(range(len(strategies)), avg_accs,
                       color=colors[:len(strategies)], edgecolor='black', linewidth=0.5)
    axes[0].set_xticks(range(len(strategies)))
    axes[0].set_xticklabels(strategies, fontsize=11)
    axes[0].set_ylabel('Average Accuracy', fontsize=12)
    axes[0].set_title('Overall Accuracy by Strategy', fontsize=14)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, acc in zip(bars, avg_accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Panel 2: Per-domain accuracy
    domain_ids = sorted(results_dict[strategies[0]]['per_domain'].keys())
    x = np.arange(len(domain_ids))
    width = 0.8 / len(strategies)

    for i, strategy in enumerate(strategies):
        domain_accs = [results_dict[strategy]['per_domain'][d] for d in domain_ids]
        axes[1].bar(x + i * width, domain_accs, width,
                    label=strategy, color=colors[i], edgecolor='black', linewidth=0.3)

    axes[1].set_xticks(x + width * (len(strategies) - 1) / 2)
    axes[1].set_xticklabels([f'Domain {d}' for d in domain_ids], fontsize=10)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Per-Domain Accuracy by Strategy', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.close()
