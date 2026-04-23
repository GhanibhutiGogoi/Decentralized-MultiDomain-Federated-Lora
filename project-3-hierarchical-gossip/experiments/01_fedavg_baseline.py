"""
Experiment 1: FedAvg Baseline with LoRA on CIFAR-100.

This establishes the centralized upper bound. Our gossip-based approach
should aim to approach this performance.

Outputs (saved under ./results/):
    - fedavg_history.json           full per-round metrics
    - fedavg_summary.json           headline numbers (final/best/fairness)
    - data_distribution.png         non-IID visualization
    - fedavg_convergence.png        avg accuracy vs round
    - fedavg_per_domain.png         per-domain accuracy curves
    - fedavg_final_per_domain.png   final per-domain bar chart
    - fedavg_summary_bars.png       final vs best accuracy bar chart

Usage:
    cd project-3-hierarchical-gossip
    python -m experiments.01_fedavg_baseline
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
from src.federated.fedavg import FedAvgServer
from src.utils.metrics import MetricsTracker, compute_fairness_metrics
from src.utils.visualization import (
    plot_training_curves,
    plot_per_domain_accuracy,
    plot_data_distribution,
    plot_fairness_bars,
    plot_final_accuracy_bars,
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
    clients_data, test_dataset = create_federated_datasets(
        data_dir=config["data"]["data_dir"],
        n_domains=config["data"]["n_domains"],
        clients_per_domain=config["data"]["clients_per_domain"],
        dirichlet_alpha=config["data"]["dirichlet_alpha"],
        batch_size=config["data"]["batch_size"],
        seed=config["training"]["seed"],
    )
    print_data_summary(clients_data)
    plot_data_distribution(clients_data, save_path=os.path.join(RESULTS_DIR, "data_distribution.png"))

    # 2. Model
    print("[Step 2] Creating LoRA ResNet-18 model...")
    global_model = create_lora_resnet(
        num_classes=config["model"]["num_classes"],
        rank=config["model"]["lora_rank"],
        alpha=config["model"]["lora_alpha"],
        pretrained=config["model"]["pretrained"],
        device=device,
    )
    trainable, total = get_trainable_param_count(global_model)
    print(f"  Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # 3. Clients
    print("[Step 3] Creating federated clients...")
    clients = []
    for client_id, data in clients_data.items():
        model = clone_model(global_model, device)
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

    # 4. FedAvg
    n_rounds = config["training"]["n_rounds"]
    print(f"\n[Step 4] Running FedAvg for {n_rounds} rounds...")
    server = FedAvgServer(clients, global_model, device=device)
    history = server.run(n_rounds=n_rounds)

    # 5. Results
    final_acc = history["avg_accuracy"][-1]
    best_acc = max(history["avg_accuracy"])
    final_domain_acc = history["per_domain_accuracy"][-1]
    fairness = compute_fairness_metrics(final_domain_acc)

    print("\n" + "=" * 60)
    print("FINAL RESULTS - FedAvg Baseline")
    print("=" * 60)
    print(f"  Final accuracy: {final_acc:.4f}")
    print(f"  Best accuracy:  {best_acc:.4f}")
    print("\n  Per-domain accuracy:")
    for did, acc in sorted(final_domain_acc.items()):
        print(f"    Domain {did} ({DOMAIN_NAMES[did]}): {acc:.4f}")
    print("\n  Fairness metrics:")
    print(f"    Accuracy gap:      {fairness['accuracy_gap']:.4f}")
    print(f"    Accuracy variance: {fairness['accuracy_variance']:.6f}")
    print("=" * 60)

    # 6. Persist raw history + summary (JSON) and plots (PNG)
    tracker = MetricsTracker("fedavg_baseline", log_dir="./logs")
    for i, r in enumerate(history["rounds"]):
        tracker.log_round(
            r,
            avg_accuracy=history["avg_accuracy"][i],
            per_domain_accuracy=history["per_domain_accuracy"][i],
        )
    tracker.save()

    with open(os.path.join(RESULTS_DIR, "fedavg_history.json"), "w") as f:
        json.dump(_to_jsonable(history), f, indent=2)
    with open(os.path.join(RESULTS_DIR, "fedavg_summary.json"), "w") as f:
        json.dump(_to_jsonable({
            "final_accuracy": final_acc,
            "best_accuracy": best_acc,
            "final_per_domain": final_domain_acc,
            "fairness": fairness,
            "n_rounds": n_rounds,
            "config": config,
        }), f, indent=2)

    plot_training_curves(
        [history], ["FedAvg"],
        save_path=os.path.join(RESULTS_DIR, "fedavg_convergence.png"),
        title="FedAvg Baseline Convergence",
    )
    plot_per_domain_accuracy(
        history, DOMAIN_NAMES,
        save_path=os.path.join(RESULTS_DIR, "fedavg_per_domain.png"),
        title="FedAvg Per-Domain Accuracy",
    )
    plot_fairness_bars(
        final_domain_acc, DOMAIN_NAMES,
        save_path=os.path.join(RESULTS_DIR, "fedavg_final_per_domain.png"),
        title="FedAvg Final Per-Domain Accuracy",
    )
    plot_final_accuracy_bars(
        [{"label": "FedAvg", "final": final_acc, "best": best_acc}],
        save_path=os.path.join(RESULTS_DIR, "fedavg_summary_bars.png"),
        title="FedAvg Final vs Best",
    )

    print(f"\nAll results saved under {RESULTS_DIR}/")
    return history


if __name__ == "__main__":
    main()
