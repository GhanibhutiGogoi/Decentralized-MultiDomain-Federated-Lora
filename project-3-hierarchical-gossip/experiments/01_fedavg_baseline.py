"""
Experiment 1: FedAvg Baseline with LoRA on CIFAR-100.

This establishes the centralized upper bound. Our gossip-based approach
should aim to approach this performance.

Usage:
    cd project-3-hierarchical-gossip
    python -m experiments.01_fedavg_baseline
"""

import sys
import os
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cifar100_domains import create_federated_datasets, print_data_summary
from src.models.lora_resnet import create_lora_resnet, get_trainable_param_count, clone_model
from src.federated.client import FederatedClient
from src.federated.fedavg import FedAvgServer
from src.utils.metrics import MetricsTracker, compute_fairness_metrics
from src.utils.visualization import plot_training_curves, plot_per_domain_accuracy
from src.data.cifar100_domains import DOMAIN_NAMES


def main():
    # Load config
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'configs', 'default_config.yaml'
    )
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set seed
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

    # 2. Create global model
    print("[Step 2] Creating LoRA ResNet-18 model...")
    global_model = create_lora_resnet(
        num_classes=config['model']['num_classes'],
        rank=config['model']['lora_rank'],
        alpha=config['model']['lora_alpha'],
        pretrained=config['model']['pretrained'],
        device=device,
    )
    trainable, total = get_trainable_param_count(global_model)
    print(f"  Trainable params: {trainable:,} / {total:,} "
          f"({100*trainable/total:.2f}%)")

    # 3. Create federated clients
    print("[Step 3] Creating federated clients...")
    clients = []
    for client_id, data in clients_data.items():
        model = clone_model(global_model, device)
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

    # 4. Run FedAvg
    print(f"\n[Step 4] Running FedAvg for {config['training']['n_rounds']} rounds...")
    server = FedAvgServer(clients, global_model, device=device)
    history = server.run(n_rounds=config['training']['n_rounds'])

    # 5. Results
    print("\n" + "=" * 60)
    print("FINAL RESULTS - FedAvg Baseline")
    print("=" * 60)

    final_acc = history['avg_accuracy'][-1]
    best_acc = max(history['avg_accuracy'])
    print(f"  Final accuracy: {final_acc:.4f}")
    print(f"  Best accuracy:  {best_acc:.4f}")

    final_domain_acc = history['per_domain_accuracy'][-1]
    fairness = compute_fairness_metrics(final_domain_acc)
    print(f"\n  Per-domain accuracy:")
    for did, acc in sorted(final_domain_acc.items()):
        print(f"    Domain {did} ({DOMAIN_NAMES[did]}): {acc:.4f}")
    print(f"\n  Fairness metrics:")
    print(f"    Accuracy gap: {fairness['accuracy_gap']:.4f}")
    print(f"    Accuracy variance: {fairness['accuracy_variance']:.6f}")
    print("=" * 60)

    # 6. Save and visualize
    tracker = MetricsTracker("fedavg_baseline", log_dir="./logs")
    for i, r in enumerate(history['rounds']):
        tracker.log_round(r,
                         avg_accuracy=history['avg_accuracy'][i],
                         per_domain_accuracy=history['per_domain_accuracy'][i])
    tracker.save()

    plot_training_curves([history], ["FedAvg"], save_path="./logs/fedavg_convergence.png")
    plot_per_domain_accuracy(history, DOMAIN_NAMES, save_path="./logs/fedavg_per_domain.png")

    return history


if __name__ == "__main__":
    main()
