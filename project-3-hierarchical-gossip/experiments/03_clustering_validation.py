"""
Experiment 3: Validate Domain Clustering on CIFAR-100.

Trains clients for a few rounds, then clusters them by LoRA similarity.
Validates that the clustering correctly recovers the true domain structure.

Usage:
    cd project-3-hierarchical-gossip
    python -m experiments.03_clustering_validation
"""

import sys
import os
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cifar100_domains import create_federated_datasets, print_data_summary, DOMAIN_NAMES
from src.models.lora_resnet import create_lora_resnet, get_trainable_param_count, clone_model
from src.federated.client import FederatedClient
from src.clustering.domain_clustering import (
    cluster_clients,
    evaluate_clustering,
    print_clustering_results,
    extract_lora_features,
)
from src.utils.visualization import plot_clustering_tsne


def main():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'configs', 'default_config.yaml'
    )
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.manual_seed(config['training']['seed'])

    # 1. Create datasets
    print("\n[Step 1] Creating federated datasets...")
    clients_data, _ = create_federated_datasets(
        data_dir=config['data']['data_dir'],
        n_domains=config['data']['n_domains'],
        clients_per_domain=config['data']['clients_per_domain'],
        dirichlet_alpha=config['data']['dirichlet_alpha'],
        batch_size=config['data']['batch_size'],
        seed=config['training']['seed'],
    )
    print_data_summary(clients_data)

    # 2. Create clients with independent models
    print("[Step 2] Creating clients...")
    base_model = create_lora_resnet(
        num_classes=config['model']['num_classes'],
        rank=config['model']['lora_rank'],
        alpha=config['model']['lora_alpha'],
        pretrained=config['model']['pretrained'],
        device=device,
    )

    clients = []
    true_domains = {}
    for client_id, data in clients_data.items():
        model = clone_model(base_model, device)
        client = FederatedClient(
            client_id=client_id,
            model=model,
            train_loader=data['train_loader'],
            test_loader=data['test_loader'],
            domain_id=data['domain_id'],
            lr=config['training']['learning_rate'],
            local_epochs=config['training']['local_epochs'],
            device=device,
        )
        clients.append(client)
        true_domains[client_id] = data['domain_id']

    # 3. Train locally (no aggregation) to let LoRA diverge by domain
    n_local_rounds = 10
    print(f"\n[Step 3] Training locally for {n_local_rounds} rounds (no aggregation)...")
    for round_idx in range(n_local_rounds):
        for client in clients:
            metrics = client.train()
        if (round_idx + 1) % 5 == 0:
            print(f"  Round {round_idx + 1}/{n_local_rounds} complete")

    # 4. Extract LoRA states and cluster
    print("\n[Step 4] Clustering clients by LoRA similarity...")
    client_lora_states = {}
    for client in clients:
        client_lora_states[client.client_id] = client.get_lora_state()

    clusters, assignments, features = cluster_clients(
        client_lora_states,
        n_clusters=config['clustering']['n_clusters'],
        method=config['clustering']['method'],
        linkage=config['clustering']['linkage'],
    )

    # 5. Evaluate clustering
    eval_metrics = evaluate_clustering(assignments, true_domains, features)
    print_clustering_results(clusters, assignments, true_domains, eval_metrics)

    # 6. Visualize
    client_ids = sorted(assignments.keys())
    true_labels = [true_domains[cid] for cid in client_ids]
    pred_labels = [assignments[cid] for cid in client_ids]

    plot_clustering_tsne(
        features, true_labels, pred_labels,
        save_path="./logs/clustering_tsne.png"
    )

    # 7. Test at different training stages
    print("\n[Step 5] Testing clustering at different training stages...")
    stages = [5, 10, 20]
    print(f"\n  Already tested at round {n_local_rounds}:")
    print(f"    ARI = {eval_metrics['adjusted_rand_index']:.4f}, "
          f"NMI = {eval_metrics['normalized_mutual_info']:.4f}")

    # Train more and re-cluster
    for target_round in stages:
        if target_round <= n_local_rounds:
            continue
        additional = target_round - n_local_rounds
        for _ in range(additional):
            for client in clients:
                client.train()
        n_local_rounds = target_round

        # Re-cluster
        client_lora_states = {}
        for client in clients:
            client_lora_states[client.client_id] = client.get_lora_state()

        _, assignments, features = cluster_clients(
            client_lora_states,
            n_clusters=config['clustering']['n_clusters'],
        )
        metrics = evaluate_clustering(assignments, true_domains, features)

        print(f"\n  After {target_round} local rounds:")
        print(f"    ARI = {metrics['adjusted_rand_index']:.4f}, "
              f"NMI = {metrics['normalized_mutual_info']:.4f}")

    print("\n" + "=" * 60)
    print("Clustering validation complete.")
    print("If ARI > 0.7, domain structure is being captured by LoRA features.")
    print("=" * 60)


if __name__ == "__main__":
    main()
