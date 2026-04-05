"""
Domain clustering based on LoRA matrix similarity.

Clusters federated learning clients into domain groups by analyzing
the singular value decomposition of their LoRA adaptation matrices.
Clients in the same domain learn similar LoRA patterns.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize


def extract_lora_features(client_lora_states):
    """
    Extract feature vectors from LoRA matrices using SVD.

    For each client, we take the singular values of A and B matrices
    as a compact representation of what the client has learned.

    Args:
        client_lora_states: dict mapping client_id -> lora_state
            where lora_state = {layer_name: {'A': tensor, 'B': tensor}}

    Returns:
        features: np.array of shape (n_clients, feature_dim)
        client_ids: list of client_ids in order
    """
    features = []
    client_ids = []

    for client_id, lora_state in client_lora_states.items():
        client_features = []

        for layer_name, params in lora_state.items():
            A = params['A'].numpy()
            B = params['B'].numpy()

            # SVD of A and B
            _, s_a, _ = np.linalg.svd(A, full_matrices=False)
            _, s_b, _ = np.linalg.svd(B, full_matrices=False)

            # Also compute Frobenius norm and spectral properties
            frob_a = np.linalg.norm(A, 'fro')
            frob_b = np.linalg.norm(B, 'fro')

            # Product BA gives the effective LoRA update
            BA = B @ A
            _, s_ba, _ = np.linalg.svd(BA, full_matrices=False)

            # Concatenate all features
            client_features.extend(s_a.tolist())
            client_features.extend(s_b.tolist())
            client_features.extend(s_ba[:min(len(s_ba), 10)].tolist())
            client_features.extend([frob_a, frob_b])

        features.append(client_features)
        client_ids.append(client_id)

    features = np.array(features)

    # Normalize features
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-10
    features = (features - mean) / std

    return features, client_ids


def cluster_clients(
    client_lora_states,
    n_clusters=5,
    method='agglomerative',
    linkage='ward',
):
    """
    Cluster clients into domain groups based on LoRA similarity.

    Args:
        client_lora_states: dict mapping client_id -> lora_state
        n_clusters: number of clusters (domains) to find
        method: clustering algorithm ('agglomerative')
        linkage: linkage criterion for agglomerative clustering

    Returns:
        clusters: dict mapping cluster_id -> list of client_ids
        assignments: dict mapping client_id -> cluster_id
        features: the feature matrix used for clustering
    """
    features, client_ids = extract_lora_features(client_lora_states)

    if method == 'agglomerative':
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
        )
        labels = clustering.fit_predict(features)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    # Build cluster dict
    clusters = {}
    assignments = {}
    for client_id, label in zip(client_ids, labels):
        label = int(label)
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(client_id)
        assignments[client_id] = label

    return clusters, assignments, features


def evaluate_clustering(predicted_assignments, true_domains, features=None):
    """
    Evaluate clustering quality against true domain labels.

    Args:
        predicted_assignments: dict mapping client_id -> predicted_cluster
        true_domains: dict mapping client_id -> true_domain_id
        features: optional feature matrix for silhouette score

    Returns:
        dict with ARI, NMI, and optionally silhouette score
    """
    # Align client ordering
    client_ids = sorted(predicted_assignments.keys())
    pred_labels = [predicted_assignments[cid] for cid in client_ids]
    true_labels = [true_domains[cid] for cid in client_ids]

    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)

    result = {
        'adjusted_rand_index': ari,
        'normalized_mutual_info': nmi,
    }

    if features is not None and len(set(pred_labels)) > 1:
        sil = silhouette_score(features, pred_labels)
        result['silhouette_score'] = sil

    return result


def print_clustering_results(clusters, assignments, true_domains, eval_metrics):
    """Pretty-print clustering results."""
    print("\n" + "=" * 60)
    print("Domain Clustering Results")
    print("=" * 60)

    for cluster_id, client_ids in sorted(clusters.items()):
        true_labels = [true_domains[cid] for cid in client_ids]
        print(f"\n  Cluster {cluster_id}: {client_ids}")
        print(f"    True domains: {true_labels}")

    print(f"\n  Metrics:")
    print(f"    Adjusted Rand Index (ARI): {eval_metrics['adjusted_rand_index']:.4f}")
    print(f"    Normalized Mutual Info (NMI): {eval_metrics['normalized_mutual_info']:.4f}")
    if 'silhouette_score' in eval_metrics:
        print(f"    Silhouette Score: {eval_metrics['silhouette_score']:.4f}")
    print("=" * 60 + "\n")
