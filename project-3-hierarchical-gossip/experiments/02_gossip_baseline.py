"""
Experiment 2: Basic Gossip Protocol (Uniform Averaging) on CIFAR-100.

This is the decentralized baseline using standard gossip without domain
awareness. Our hierarchical gossip should outperform this.

Outputs (saved under ./results/):
    - gossip_history.json              full per-round metrics
    - gossip_summary.json              headline numbers + topology info
    - gossip_convergence.png           avg accuracy vs round
    - gossip_per_domain.png            per-domain accuracy curves
    - gossip_final_per_domain.png      final per-domain bar chart
    - gossip_communication_cost.png    cumulative gossip messages
    - gossip_vs_fedavg_convergence.png (if fedavg_history.json is present)
    - gossip_vs_fedavg_summary.png     (if fedavg_summary.json is present)

Usage:
    cd project-3-hierarchical-gossip
    python -m experiments.02_gossip_baseline
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
from src.federated.gossip import GossipProtocol
from src.utils.metrics import MetricsTracker, compute_fairness_metrics
from src.utils.visualization import (
    plot_training_curves,
    plot_per_domain_accuracy,
    plot_fairness_bars,
    plot_communication_cost,
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

    # 2. Model
    print("[Step 2] Creating LoRA ResNet-18 model...")
    base_model = create_lora_resnet(
        num_classes=config["model"]["num_classes"],
        rank=config["model"]["lora_rank"],
        alpha=config["model"]["lora_alpha"],
        pretrained=config["model"]["pretrained"],
        device=device,
    )
    trainable, total = get_trainable_param_count(base_model)
    print(f"  Trainable params: {trainable:,} / {total:,}")

    # 3. Clients
    print("[Step 3] Creating federated clients...")
    clients = []
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

    # 4. Gossip
    topology = config["gossip"]["topology"]
    n_rounds = config["training"]["n_rounds"]
    print(f"\n[Step 4] Running Gossip ({topology} topology) for {n_rounds} rounds...")

    gossip = GossipProtocol(clients, topology=topology, seed=config["training"]["seed"])
    topo_info = gossip.get_topology_info()
    print(f"  Topology: {topo_info['n_clients']} clients, {topo_info['n_edges']} edges, "
          f"avg degree: {topo_info['avg_degree']:.1f}")

    history = gossip.run(n_rounds=n_rounds)

    # 5. Results
    final_acc = history["avg_accuracy"][-1]
    best_acc = max(history["avg_accuracy"])
    total_messages = sum(history["messages_per_round"])
    final_domain_acc = history["per_domain_accuracy"][-1]
    fairness = compute_fairness_metrics(final_domain_acc)

    print("\n" + "=" * 60)
    print(f"FINAL RESULTS - Gossip Baseline ({topology})")
    print("=" * 60)
    print(f"  Final accuracy: {final_acc:.4f}")
    print(f"  Best accuracy:  {best_acc:.4f}")
    print(f"  Total messages: {total_messages}")
    print("\n  Per-domain accuracy:")
    for did, acc in sorted(final_domain_acc.items()):
        print(f"    Domain {did} ({DOMAIN_NAMES[did]}): {acc:.4f}")
    print("\n  Fairness metrics:")
    print(f"    Accuracy gap: {fairness['accuracy_gap']:.4f}")
    print("=" * 60)

    # 6. Persist raw history + summary (JSON) and plots (PNG)
    tracker = MetricsTracker("gossip_baseline", log_dir="./logs")
    for i, r in enumerate(history["rounds"]):
        tracker.log_round(
            r,
            avg_accuracy=history["avg_accuracy"][i],
            per_domain_accuracy=history["per_domain_accuracy"][i],
            messages=history["messages_per_round"][i],
        )
    tracker.save()

    with open(os.path.join(RESULTS_DIR, "gossip_history.json"), "w") as f:
        json.dump(_to_jsonable(history), f, indent=2)
    with open(os.path.join(RESULTS_DIR, "gossip_summary.json"), "w") as f:
        json.dump(_to_jsonable({
            "final_accuracy": final_acc,
            "best_accuracy": best_acc,
            "total_messages": total_messages,
            "topology": topology,
            "topology_info": topo_info,
            "final_per_domain": final_domain_acc,
            "fairness": fairness,
            "n_rounds": n_rounds,
            "config": config,
        }), f, indent=2)

    plot_training_curves(
        [history], [f"Gossip ({topology})"],
        save_path=os.path.join(RESULTS_DIR, "gossip_convergence.png"),
        title=f"Gossip Baseline Convergence ({topology})",
    )
    plot_per_domain_accuracy(
        history, DOMAIN_NAMES,
        save_path=os.path.join(RESULTS_DIR, "gossip_per_domain.png"),
        title=f"Gossip Per-Domain Accuracy ({topology})",
    )
    plot_fairness_bars(
        final_domain_acc, DOMAIN_NAMES,
        save_path=os.path.join(RESULTS_DIR, "gossip_final_per_domain.png"),
        title=f"Gossip Final Per-Domain Accuracy ({topology})",
    )
    plot_communication_cost(
        history,
        save_path=os.path.join(RESULTS_DIR, "gossip_communication_cost.png"),
        title="Cumulative Gossip Messages",
    )

    # 7. If FedAvg results exist, overlay a comparison
    fedavg_history_path = os.path.join(RESULTS_DIR, "fedavg_history.json")
    fedavg_summary_path = os.path.join(RESULTS_DIR, "fedavg_summary.json")
    if os.path.exists(fedavg_history_path):
        with open(fedavg_history_path) as f:
            fedavg_history = json.load(f)
        plot_training_curves(
            [fedavg_history, history],
            ["FedAvg", f"Gossip ({topology})"],
            save_path=os.path.join(RESULTS_DIR, "gossip_vs_fedavg_convergence.png"),
            title="FedAvg vs Gossip Convergence",
        )

    if os.path.exists(fedavg_summary_path):
        with open(fedavg_summary_path) as f:
            fedavg_summary = json.load(f)
        plot_final_accuracy_bars(
            [
                {"label": "FedAvg",
                 "final": fedavg_summary["final_accuracy"],
                 "best": fedavg_summary["best_accuracy"]},
                {"label": f"Gossip ({topology})",
                 "final": final_acc, "best": best_acc},
            ],
            save_path=os.path.join(RESULTS_DIR, "gossip_vs_fedavg_summary.png"),
            title="Final vs Best Accuracy — FedAvg vs Gossip",
        )

    print(f"\nAll results saved under {RESULTS_DIR}/")
    return history


if __name__ == "__main__":
    main()
