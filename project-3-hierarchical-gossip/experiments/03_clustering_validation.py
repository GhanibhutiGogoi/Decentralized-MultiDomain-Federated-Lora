"""
Experiment 3: Validate Domain Clustering on CIFAR-100.

Trains clients locally (no aggregation) so each client's LoRA diverges
toward its own domain, then clusters them by LoRA similarity and
evaluates how well the clustering recovers true domain structure at
several training stages.

Outputs (saved under ./results/):
    - clustering_results.json                per-stage ARI/NMI/silhouette
    - clustering_tsne.png                    t-SNE at final stage
    - clustering_confusion_matrix.png        true-domain x predicted-cluster
    - clustering_metrics_over_rounds.png     ARI/NMI trajectory

Usage:
    cd project-3-hierarchical-gossip
    python -m experiments.03_clustering_validation
"""

import sys
import os
import json
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cifar100_domains import (
    create_federated_datasets,
    print_data_summary,
    DOMAIN_NAMES,
)
from src.models.lora_resnet import (
    create_lora_resnet,
    get_trainable_param_count,
    clone_model,
)
from src.federated.client import FederatedClient
from src.clustering.domain_clustering import (
    cluster_clients,
    evaluate_clustering,
    print_clustering_results,
)
from src.utils.visualization import (
    plot_clustering_tsne,
    plot_confusion_matrix,
    plot_clustering_metrics_over_rounds,
)


RESULTS_DIR = "./results"


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return obj


def _cluster_and_eval(clients, true_domains, n_clusters, method, linkage):
    client_lora_states = {c.client_id: c.get_lora_state() for c in clients}
    clusters, assignments, features = cluster_clients(
        client_lora_states,
        n_clusters=n_clusters,
        method=method,
        linkage=linkage,
    )
    metrics = evaluate_clustering(assignments, true_domains, features)
    return clusters, assignments, features, metrics


def main():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "default_config.yaml",
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    torch.manual_seed(config["training"]["seed"])

    # 1. Data
    print("\n[Step 1] Creating federated datasets...")
    clients_data, _ = create_federated_datasets(
        data_dir=config["data"]["data_dir"],
        n_domains=config["data"]["n_domains"],
        clients_per_domain=config["data"]["clients_per_domain"],
        dirichlet_alpha=config["data"]["dirichlet_alpha"],
        batch_size=config["data"]["batch_size"],
        seed=config["training"]["seed"],
    )
    print_data_summary(clients_data)

    # 2. Model & clients
    print("[Step 2] Creating clients...")
    base_model = create_lora_resnet(
        num_classes=config["model"]["num_classes"],
        rank=config["model"]["lora_rank"],
        alpha=config["model"]["lora_alpha"],
        pretrained=config["model"]["pretrained"],
        device=device,
    )

    clients = []
    true_domains = {}
    for client_id, data in clients_data.items():
        model = clone_model(base_model, device)
        client = FederatedClient(
            client_id=client_id,
            model=model,
            train_loader=data["train_loader"],
            test_loader=data["test_loader"],
            domain_id=data["domain_id"],
            lr=config["training"]["learning_rate"],
            local_epochs=config["training"]["local_epochs"],
            device=device,
        )
        clients.append(client)
        true_domains[client_id] = data["domain_id"]

    # 3. Train incrementally through a schedule of stages and re-cluster at each
    stages = [2, 5, 10, 20]
    print(f"\n[Step 3] Local-only training across stages: {stages}")
    n_clusters = config["clustering"]["n_clusters"]
    method = config["clustering"]["method"]
    linkage = config["clustering"]["linkage"]

    stage_results = []
    prev = 0
    last_features = None
    last_assignments = None
    last_clusters = None

    for stage in stages:
        delta = stage - prev
        if delta > 0:
            print(f"\n  -> training {delta} more round(s) (now at {stage})...")
            for _ in range(delta):
                for client in clients:
                    client.train()
        clusters, assignments, features, metrics = _cluster_and_eval(
            clients, true_domains, n_clusters, method, linkage
        )
        print(f"  Stage {stage:>3} rounds | ARI = {metrics['adjusted_rand_index']:.4f} | "
              f"NMI = {metrics['normalized_mutual_info']:.4f}"
              + (f" | Silhouette = {metrics['silhouette_score']:.4f}"
                 if "silhouette_score" in metrics else ""))
        stage_results.append({"stage": stage, **metrics})
        prev = stage
        last_features = features
        last_assignments = assignments
        last_clusters = clusters

    # Detailed pretty-print for the final stage
    print_clustering_results(last_clusters, last_assignments, true_domains, stage_results[-1])

    # 4. Save JSON
    with open(os.path.join(RESULTS_DIR, "clustering_results.json"), "w") as f:
        json.dump(_to_jsonable({
            "stages": stage_results,
            "final_stage": stages[-1],
            "final_assignments": last_assignments,
            "true_domains": true_domains,
            "config": {
                "n_clusters": n_clusters,
                "method": method,
                "linkage": linkage,
                "seed": config["training"]["seed"],
            },
        }), f, indent=2)

    # 5. Plots
    client_ids = sorted(last_assignments.keys())
    true_labels = [true_domains[cid] for cid in client_ids]
    pred_labels = [last_assignments[cid] for cid in client_ids]

    plot_clustering_tsne(
        last_features, true_labels, pred_labels,
        save_path=os.path.join(RESULTS_DIR, "clustering_tsne.png"),
    )
    plot_confusion_matrix(
        true_labels, pred_labels,
        row_names=[DOMAIN_NAMES[d] for d in sorted(set(true_labels))],
        save_path=os.path.join(RESULTS_DIR, "clustering_confusion_matrix.png"),
        title=f"Clustering Confusion Matrix (after {stages[-1]} local rounds)",
    )
    plot_clustering_metrics_over_rounds(
        stages=[r["stage"] for r in stage_results],
        ari_values=[r["adjusted_rand_index"] for r in stage_results],
        nmi_values=[r["normalized_mutual_info"] for r in stage_results],
        silhouette_values=[r.get("silhouette_score") for r in stage_results],
        save_path=os.path.join(RESULTS_DIR, "clustering_metrics_over_rounds.png"),
    )

    print("\n" + "=" * 60)
    print("Clustering validation complete.")
    print(f"  Final ARI = {stage_results[-1]['adjusted_rand_index']:.4f}  "
          f"(> 0.7 means domain structure is being captured by LoRA features).")
    print(f"  Results saved under {RESULTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
