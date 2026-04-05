"""
Experiment 2: Basic Gossip Protocol (Uniform Averaging) on CIFAR-100.

This is the decentralized baseline using standard gossip without
domain awareness. Our hierarchical gossip should outperform this.

Usage:
    cd project-3-hierarchical-gossip
    python -m experiments.02_gossip_baseline
"""

import sys
import os
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cifar100_domains import create_federated_datasets, print_data_summary, DOMAIN_NAMES
from src.models.lora_resnet import create_lora_resnet, get_trainable_param_count, clone_model
from src.federated.client import FederatedClient
from src.federated.gossip import GossipProtocol
from src.utils.metrics import MetricsTracker, compute_fairness_metrics
from src.utils.visualization import plot_training_curves, plot_per_domain_accuracy


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

    # 1. Create federated datasets
    print("\n[Step 1] Creating federated datasets...")
    clients_data, test_dataset = create_federated_datasets(
        data_dir=config['data']['data_dir'],
        n_domains=config['data']['n_domains'],
        clients_per_domain=config['data']['clients_per_domain'],
        dirichlet_alpha=config['data']['dirichlet_alpha'],
        batch_size=config['data']['batch_size'],
        seed=config['training']['seed'],
    )
    print_data_summary(clients_data)

    # 2. Create base model
    print("[Step 2] Creating LoRA ResNet-18 model...")
    base_model = create_lora_resnet(
        num_classes=config['model']['num_classes'],
        rank=config['model']['lora_rank'],
        alpha=config['model']['lora_alpha'],
        pretrained=config['model']['pretrained'],
        device=device,
    )
    trainable, total = get_trainable_param_count(base_model)
    print(f"  Trainable params: {trainable:,} / {total:,}")

    # 3. Create federated clients
    print("[Step 3] Creating federated clients...")
    clients = []
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

    # 4. Run Gossip
    topology = config['gossip']['topology']
    print(f"\n[Step 4] Running Gossip ({topology} topology) for {config['training']['n_rounds']} rounds...")

    gossip = GossipProtocol(clients, topology=topology, seed=config['training']['seed'])
    topo_info = gossip.get_topology_info()
    print(f"  Topology: {topo_info['n_clients']} clients, {topo_info['n_edges']} edges, "
          f"avg degree: {topo_info['avg_degree']:.1f}")

    history = gossip.run(n_rounds=config['training']['n_rounds'])

    # 5. Results
    print("\n" + "=" * 60)
    print(f"FINAL RESULTS - Gossip Baseline ({topology})")
    print("=" * 60)

    final_acc = history['avg_accuracy'][-1]
    best_acc = max(history['avg_accuracy'])
    total_messages = sum(history['messages_per_round'])

    print(f"  Final accuracy: {final_acc:.4f}")
    print(f"  Best accuracy:  {best_acc:.4f}")
    print(f"  Total messages: {total_messages}")

    final_domain_acc = history['per_domain_accuracy'][-1]
    fairness = compute_fairness_metrics(final_domain_acc)
    print(f"\n  Per-domain accuracy:")
    for did, acc in sorted(final_domain_acc.items()):
        print(f"    Domain {did} ({DOMAIN_NAMES[did]}): {acc:.4f}")
    print(f"\n  Fairness metrics:")
    print(f"    Accuracy gap: {fairness['accuracy_gap']:.4f}")
    print("=" * 60)

    # 6. Save
    tracker = MetricsTracker("gossip_baseline", log_dir="./logs")
    for i, r in enumerate(history['rounds']):
        tracker.log_round(r,
                         avg_accuracy=history['avg_accuracy'][i],
                         per_domain_accuracy=history['per_domain_accuracy'][i],
                         messages=history['messages_per_round'][i])
    tracker.save()

    plot_training_curves([history], [f"Gossip ({topology})"],
                        save_path="./logs/gossip_convergence.png")
    plot_per_domain_accuracy(history, DOMAIN_NAMES,
                            save_path="./logs/gossip_per_domain.png")

    return history


if __name__ == "__main__":
    main()
