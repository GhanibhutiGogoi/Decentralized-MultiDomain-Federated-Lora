"""
Visualization utilities for federated learning experiments.

All plots use a non-interactive Matplotlib backend so scripts run headlessly
(no plt.show() blocking, safe in SSH/CI/background).
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


def _ensure_parent(save_path):
    if save_path:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)


def _finalize(fig, save_path):
    fig.tight_layout()
    if save_path:
        _ensure_parent(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.close(fig)


def plot_training_curves(histories, labels, save_path=None, title="Federated Learning Convergence"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for history, label in zip(histories, labels):
        ax.plot(history["rounds"], history["avg_accuracy"], label=label, linewidth=2)
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Average Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    _finalize(fig, save_path)


def plot_per_domain_accuracy(history, domain_names=None, save_path=None, title="Per-Domain Accuracy"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    rounds = history["rounds"]
    domain_ids = sorted(history["per_domain_accuracy"][0].keys())
    for did in domain_ids:
        accs = [r[did] for r in history["per_domain_accuracy"]]
        label = domain_names[did] if domain_names else f"Domain {did}"
        ax.plot(rounds, accs, label=label, linewidth=2)
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    _finalize(fig, save_path)


def plot_clustering_tsne(features, true_labels, predicted_labels, save_path=None):
    if features.shape[0] < 4:
        perplexity = max(1, features.shape[0] - 1)
    else:
        perplexity = min(30, features.shape[0] - 1)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
    coords = tsne.fit_transform(features)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    s1 = axes[0].scatter(coords[:, 0], coords[:, 1], c=true_labels, cmap="tab10",
                         s=120, edgecolors="black", linewidth=0.5)
    axes[0].set_title("True Domain Labels", fontsize=13)
    axes[0].legend(*s1.legend_elements(), title="Domain", loc="best")
    s2 = axes[1].scatter(coords[:, 0], coords[:, 1], c=predicted_labels, cmap="tab10",
                         s=120, edgecolors="black", linewidth=0.5)
    axes[1].set_title("Predicted Clusters", fontsize=13)
    axes[1].legend(*s2.legend_elements(), title="Cluster", loc="best")
    for ax in axes:
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, alpha=0.25)
    _finalize(fig, save_path)


def plot_data_distribution(clients_data, save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    client_ids = sorted(clients_data.keys())
    n_samples = [clients_data[cid]["n_samples"] for cid in client_ids]
    domains = [clients_data[cid]["domain_id"] for cid in client_ids]
    max_d = max(domains) if max(domains) > 0 else 1
    colors = plt.cm.tab10(np.array(domains) / max_d)
    ax.bar(range(len(client_ids)), n_samples, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Client ID", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title("Data Distribution Across Clients", fontsize=14)
    ax.set_xticks(range(len(client_ids)))
    ax.set_xticklabels([str(cid) for cid in client_ids])
    ax.grid(True, axis="y", alpha=0.3)

    unique_domains = sorted(set(domains))
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=plt.cm.tab10(d / max_d))
                      for d in unique_domains]
    legend_labels = [
        clients_data[next(cid for cid in client_ids if clients_data[cid]["domain_id"] == d)]["domain_name"]
        for d in unique_domains
    ]
    ax.legend(legend_handles, legend_labels, title="Domain", loc="upper right")
    _finalize(fig, save_path)


def plot_fairness_bars(per_domain_accuracy, domain_names=None, save_path=None,
                       title="Final Per-Domain Accuracy"):
    """Bar chart of per-domain accuracies with mean and gap annotations."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    dids = sorted(per_domain_accuracy.keys())
    accs = [per_domain_accuracy[d] for d in dids]
    labels = [domain_names[d] if domain_names else f"D{d}" for d in dids]
    colors = plt.cm.tab10(np.linspace(0, 1, len(dids)))
    bars = ax.bar(range(len(dids)), accs, color=colors, edgecolor="black", linewidth=0.5)

    mean_acc = float(np.mean(accs))
    ax.axhline(mean_acc, color="red", linestyle="--", linewidth=1.5,
               label=f"mean = {mean_acc:.3f}")
    ax.set_xticks(range(len(dids)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Accuracy", fontsize=12)
    gap = max(accs) - min(accs)
    ax.set_title(f"{title}  (gap = {gap:.3f})", fontsize=13)
    ax.set_ylim(0, max(1.0, max(accs) * 1.15))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    _finalize(fig, save_path)


def plot_final_accuracy_bars(summaries, save_path=None, title="Final vs Best Accuracy"):
    """
    Compare multiple experiments' final / best accuracy as grouped bars.
    summaries: list of dicts each like {"label": "FedAvg", "final": 0.5, "best": 0.6}
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    labels = [s["label"] for s in summaries]
    x = np.arange(len(labels))
    width = 0.35
    finals = [s["final"] for s in summaries]
    bests = [s["best"] for s in summaries]
    ax.bar(x - width / 2, finals, width, label="Final", color="#4C72B0")
    ax.bar(x + width / 2, bests, width, label="Best", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    for xi, (f, b) in enumerate(zip(finals, bests)):
        ax.text(xi - width / 2, f + 0.01, f"{f:.3f}", ha="center", fontsize=9)
        ax.text(xi + width / 2, b + 0.01, f"{b:.3f}", ha="center", fontsize=9)
    _finalize(fig, save_path)


def plot_communication_cost(history, save_path=None, title="Cumulative Gossip Messages"):
    """Cumulative messages vs round."""
    msgs = history.get("messages_per_round", [])
    rounds = history.get("rounds", list(range(len(msgs))))
    if not msgs:
        return
    cumulative = np.cumsum(msgs)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(rounds, cumulative, linewidth=2, color="#55A868")
    ax.fill_between(rounds, 0, cumulative, alpha=0.2, color="#55A868")
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Cumulative messages", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    _finalize(fig, save_path)


def plot_confusion_matrix(true_labels, pred_labels, save_path=None,
                          title="Clustering Confusion Matrix",
                          row_names=None, col_names=None):
    """Confusion matrix heatmap of true_domain x predicted_cluster."""
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    t_u = sorted(set(true_labels.tolist()))
    p_u = sorted(set(pred_labels.tolist()))
    mat = np.zeros((len(t_u), len(p_u)), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        mat[t_u.index(t), p_u.index(p)] += 1

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    sns.heatmap(
        mat, annot=True, fmt="d", cmap="Blues",
        xticklabels=(col_names or [f"C{p}" for p in p_u]),
        yticklabels=(row_names or [f"D{t}" for t in t_u]),
        cbar=True, ax=ax,
    )
    ax.set_xlabel("Predicted Cluster", fontsize=12)
    ax.set_ylabel("True Domain", fontsize=12)
    ax.set_title(title, fontsize=13)
    _finalize(fig, save_path)


def plot_clustering_metrics_over_rounds(stages, ari_values, nmi_values, save_path=None,
                                         silhouette_values=None,
                                         title="Clustering Quality vs Training Rounds"):
    """Line plot of ARI / NMI (and optional silhouette) as local training progresses."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.plot(stages, ari_values, "o-", label="ARI", linewidth=2, markersize=8)
    ax.plot(stages, nmi_values, "s-", label="NMI", linewidth=2, markersize=8)
    if silhouette_values is not None and any(v is not None for v in silhouette_values):
        clean = [(s, v) for s, v in zip(stages, silhouette_values) if v is not None]
        if clean:
            sx, sy = zip(*clean)
            ax.plot(sx, sy, "^-", label="Silhouette", linewidth=2, markersize=8)
    ax.axhline(0.7, color="red", linestyle="--", linewidth=1, alpha=0.6,
               label="ARI target = 0.7")
    ax.set_xlabel("Local training rounds", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    _finalize(fig, save_path)
