"""
Visualization utilities for federated learning experiments.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


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
    plt.show()


def plot_per_domain_accuracy(history, domain_names=None, save_path=None):
    """
    Plot per-domain accuracy over rounds.

    Args:
        history: history dict with 'per_domain_accuracy' list
        domain_names: optional dict mapping domain_id -> name
        save_path: optional path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Collect domain accuracies
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
    plt.show()


def plot_clustering_tsne(features, true_labels, predicted_labels, save_path=None):
    """
    Plot t-SNE visualization of LoRA features with true vs predicted clusters.

    Args:
        features: (n_clients, feature_dim) array
        true_labels: list of true domain IDs
        predicted_labels: list of predicted cluster IDs
        save_path: optional path to save figure
    """
    if features.shape[0] < 4:
        perplexity = features.shape[0] - 1
    else:
        perplexity = min(30, features.shape[0] - 1)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    coords = tsne.fit_transform(features)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # True labels
    scatter1 = axes[0].scatter(
        coords[:, 0], coords[:, 1],
        c=true_labels, cmap='tab10', s=100, edgecolors='black', linewidth=0.5
    )
    axes[0].set_title('True Domain Labels', fontsize=13)
    axes[0].legend(*scatter1.legend_elements(), title="Domain")

    # Predicted labels
    scatter2 = axes[1].scatter(
        coords[:, 0], coords[:, 1],
        c=predicted_labels, cmap='tab10', s=100, edgecolors='black', linewidth=0.5
    )
    axes[1].set_title('Predicted Clusters', fontsize=13)
    axes[1].legend(*scatter2.legend_elements(), title="Cluster")

    for ax in axes:
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_data_distribution(clients_data, save_path=None):
    """
    Plot data distribution across clients showing non-IID-ness.

    Args:
        clients_data: dict from create_federated_datasets
        save_path: optional path to save
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    client_ids = sorted(clients_data.keys())
    n_samples = [clients_data[cid]['n_samples'] for cid in client_ids]
    domains = [clients_data[cid]['domain_id'] for cid in client_ids]

    colors = plt.cm.tab10(np.array(domains) / max(domains))
    bars = ax.bar(range(len(client_ids)), n_samples, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Client ID', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Data Distribution Across Clients', fontsize=14)
    ax.set_xticks(range(len(client_ids)))
    ax.set_xticklabels([str(cid) for cid in client_ids])

    # Legend for domains
    unique_domains = sorted(set(domains))
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=plt.cm.tab10(d / max(domains)))
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
    plt.show()
