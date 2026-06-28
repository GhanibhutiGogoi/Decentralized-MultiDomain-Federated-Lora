"""
Experiment 04: Allocation Strategy Comparison

Purpose:
    Compare LoRA rank allocation strategies in federated learning:
    1. Uniform: all clients get rank 16
    2. Random: each client gets a random rank
    3. Domain-Aware: rank allocated based on domain complexity score
    4. Dynamic Allocation: Gabriel-style adaptive rank output + alpha/scaling policy

    Uses HeteroFedAvg to handle mixed LoRA ranks across clients.

Outputs:
    - results/experiment_04_allocation_comparison/allocation_comparison.json
    - results/experiment_04_allocation_comparison/allocation_convergence.png
    - results/experiment_04_allocation_comparison/allocation_per_domain.png
    - Console comparison table

Note:
    Dynamic Allocation is integrated as the first bridge between
    Gabriel's adaptive rank-selection output and Project 2 alpha/scaling policy.
"""

import os
import sys
import json
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cifar100_domains import create_federated_datasets
from src.models.lora_resnet import create_lora_resnet
from src.federated.client import FederatedClient
from src.federated.hetero_fedavg import HeteroFedAvgServer
from src.allocation.rank_allocator import RankAllocator
from src.allocation.dynamic_allocation_policy import dynamic_allocation
from src.complexity.domain_complexity import DomainComplexityAnalyzer
from src.utils.metrics import compute_fairness_metrics
from src.utils.visualization import (
    plot_training_curves,
    plot_allocation_comparison,
)


def run_fedavg_with_allocation(
    clients_data,
    rank_assignments,
    config,
    device,
    alpha,
    desc="",
):
    """
    Create clients with specified rank assignments and run HeteroFedAvg.

    Args:
        clients_data: dict from create_federated_datasets
        rank_assignments: dict mapping client_id -> rank
        config: config dict
        device: compute device
        alpha: LoRA alpha value
        desc: description for progress bar

    Returns:
        history dict from HeteroFedAvgServer.run()
    """
    clients = []

    for cid in sorted(clients_data.keys()):
        rank = rank_assignments[cid]

        model = create_lora_resnet(
            num_classes=config["model"]["num_classes"],
            rank=rank,
            alpha=alpha,
            pretrained=True,
            device=device,
        )

        client = FederatedClient(
            client_id=cid,
            model=model,
            train_loader=clients_data[cid]["train_loader"],
            test_loader=clients_data[cid]["test_loader"],
            domain_id=clients_data[cid]["domain_id"],
            lr=config["training"]["learning_rate"],
            local_epochs=config["training"]["local_epochs"],
            device=device,
        )

        clients.append(client)

    server = HeteroFedAvgServer(
        clients=clients,
        rank_assignments=rank_assignments,
        alpha=alpha,
        device=device,
    )

    history = server.run(
        n_rounds=config["training"]["n_rounds"],
        verbose=True,
    )

    return history


def build_gabriel_style_rank_assignments(n_clients):
    """
    Build temporary Gabriel-style rank assignments.

    Gabriel's Project 1 configuration maps:
        batch 16  -> max rank 4
        batch 64  -> max rank 8
        batch 256 -> max rank 16

    Until Gabriel's actual per-client rank output file is integrated,
    we use this repeating 4/8/16 pattern as a structural bridge.

    Args:
        n_clients: number of federated clients

    Returns:
        dict mapping integer client_id -> selected rank
    """
    rank_pattern = [4, 8, 16]

    return {
        cid: rank_pattern[cid % len(rank_pattern)]
        for cid in range(n_clients)
    }


def build_dynamic_allocation_strategy(n_clients, alpha=64):
    """
    Build dynamic allocation using Gabriel-style rank output and alpha policy.

    The dynamic allocation policy produces:

        client_i -> rank r_i -> alpha alpha_i -> scaling alpha_i / r_i

    Returns:
        rank_assignments:
            dict mapping integer client_id -> rank, used by HeteroFedAvg

        dynamic_allocations:
            full allocation dict containing rank, alpha, and scaling
    """
    gabriel_style_ranks = build_gabriel_style_rank_assignments(n_clients)

    rank_assignments_for_policy = {
        f"client_{cid}": rank
        for cid, rank in gabriel_style_ranks.items()
    }

    dynamic_allocations = dynamic_allocation(
        rank_assignments=rank_assignments_for_policy,
        alpha=alpha,
    )

    rank_assignments = {
        int(client_id.replace("client_", "")): allocation["rank"]
        for client_id, allocation in dynamic_allocations.items()
    }

    return rank_assignments, dynamic_allocations


def load_complexity_scores(results_root, clients_data, train_dataset, config, device):
    """
    Load complexity scores from Experiment 01 if available.
    Otherwise, compute them fresh.
    """
    possible_paths = [
        os.path.join(
            results_root,
            "experiment_01_data_and_complexity",
            "complexity_scores.json",
        ),
        os.path.join(
            results_root,
            "complexity_scores.json",
        ),
    ]

    for complexity_path in possible_paths:
        if os.path.exists(complexity_path):
            print(f"\nLoading complexity scores from: {complexity_path}")
            with open(complexity_path) as f:
                complexity_data = json.load(f)

            return {
                int(k): v["score"]
                for k, v in complexity_data.items()
            }

    print("\nComputing complexity scores because Experiment 01 results were not found...")

    analyzer = DomainComplexityAnalyzer(
        device=device,
        n_feature_samples=config["complexity"]["n_feature_samples"],
    )

    complexity_results = analyzer.analyze_all_clients(
        clients_data,
        train_dataset,
    )

    return {
        cid: result["score"]
        for cid, result in complexity_results.items()
    }


def summarize_strategy_metadata(rank_assignments, alpha):
    """
    Summarize rank/alpha/scaling information for reporting.
    """
    ranks = list(rank_assignments.values())
    scalings = [
        alpha / rank
        for rank in ranks
    ]

    return {
        "num_clients": len(rank_assignments),
        "total_rank": sum(ranks),
        "average_rank": float(np.mean(ranks)),
        "alpha": alpha,
        "average_scaling": float(np.mean(scalings)),
        "rank_assignments": {
            str(k): v
            for k, v in rank_assignments.items()
        },
    }


def main():
    # Load config
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    config_path = os.path.join(
        project_root,
        "configs",
        "default_config.yaml",
    )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = config["training"]["seed"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Results directory
    results_root = config["logging"]["results_dir"]

    results_dir = os.path.join(
        results_root,
        "experiment_04_allocation_comparison",
    )

    os.makedirs(results_dir, exist_ok=True)

    # Create datasets
    print("\nCreating federated datasets...")

    clients_data, train_dataset, test_dataset = create_federated_datasets(
        data_dir=config["data"]["data_dir"],
        n_domains=config["data"]["n_domains"],
        clients_per_domain=config["data"]["clients_per_domain"],
        dirichlet_alpha=config["data"]["dirichlet_alpha"],
        batch_size=config["data"]["batch_size"],
        seed=seed,
    )

    n_clients = len(clients_data)

    # Load or compute complexity scores
    complexity_scores = load_complexity_scores(
        results_root=results_root,
        clients_data=clients_data,
        train_dataset=train_dataset,
        config=config,
        device=device,
    )

    # Alpha policy
    # Experiment 06 showed alpha=64 as a strong global alpha.
    dynamic_alpha = 64

    # Build dynamic allocation strategy
    dynamic_rank_assignments, dynamic_allocations = build_dynamic_allocation_strategy(
        n_clients=n_clients,
        alpha=dynamic_alpha,
    )

    # Define allocation strategies
    strategies = {
        "Uniform (r=16)": {
            "rank_assignments": RankAllocator.uniform(
                n_clients,
                rank=16,
            ),
            "alpha": dynamic_alpha,
            "metadata": {
                "type": "uniform",
                "description": "All clients receive rank 16 and alpha 64.",
            },
        },

        "Random": {
            "rank_assignments": RankAllocator.random(
                n_clients,
                seed=seed,
            ),
            "alpha": dynamic_alpha,
            "metadata": {
                "type": "random",
                "description": "Each client receives a random rank.",
            },
        },

        "Domain-Aware": {
            "rank_assignments": RankAllocator.domain_aware(
                complexity_scores,
                min_rank=config["allocation"]["min_rank"],
                max_rank=config["allocation"]["max_rank"],
                base_rank=config["allocation"]["base_rank"],
            ),
            "alpha": dynamic_alpha,
            "metadata": {
                "type": "domain_aware",
                "description": "Rank allocated based on domain complexity score.",
            },
        },

        "Dynamic Allocation": {
            "rank_assignments": dynamic_rank_assignments,
            "alpha": dynamic_alpha,
            "metadata": {
                "type": "dynamic_allocation",
                "description": (
                    "Gabriel-style adaptive rank output connected with "
                    "Project 2 alpha/scaling policy."
                ),
                "dynamic_allocations": dynamic_allocations,
                "note": (
                    "This uses a temporary Gabriel-style 4/8/16 rank pattern. "
                    "Future runs should replace it with Gabriel's actual "
                    "per-client rank output file."
                ),
            },
        },
    }

    # Print allocations
    print("\n" + "=" * 60)
    print("Rank Allocations:")
    print("=" * 60)

    for name, strategy in strategies.items():
        rank_assignments = strategy["rank_assignments"]
        ranks = list(rank_assignments.values())

        print(f"\n  {name}:")
        print(f"    Ranks: {ranks}")
        print(f"    Mean rank: {np.mean(ranks):.1f}")
        print(f"    Total rank: {sum(ranks)}")
        print(f"    Alpha: {strategy['alpha']}")

    print("=" * 60)

    # Run experiments
    all_histories = {}
    all_results = {}

    for name, strategy in strategies.items():
        print(f"\n{'=' * 60}")
        print(f"Running: {name}")
        print(f"{'=' * 60}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        rank_assignments = strategy["rank_assignments"]
        alpha = strategy["alpha"]

        history = run_fedavg_with_allocation(
            clients_data=clients_data,
            rank_assignments=rank_assignments,
            config=config,
            device=device,
            alpha=alpha,
            desc=name,
        )

        all_histories[name] = history

        # Final metrics
        final_acc = history["avg_accuracy"][-1]
        final_domain = history["per_domain_accuracy"][-1]
        fairness = compute_fairness_metrics(final_domain)

        allocation_metadata = summarize_strategy_metadata(
            rank_assignments=rank_assignments,
            alpha=alpha,
        )

        all_results[name] = {
            "avg_accuracy": final_acc,
            "per_domain": {
                str(k): v
                for k, v in final_domain.items()
            },
            "fairness": fairness,
            "rank_assignments": {
                str(k): v
                for k, v in rank_assignments.items()
            },
            "total_rank": sum(rank_assignments.values()),
            "alpha": alpha,
            "allocation_metadata": allocation_metadata,
            "strategy_metadata": strategy["metadata"],
        }

        print(f"\n  Final accuracy: {final_acc:.4f}")
        print(f"  Fairness gap: {fairness['accuracy_gap']:.4f}")

    # Comparison Summary
    print("\n" + "=" * 80)
    print("ALLOCATION STRATEGY COMPARISON")
    print("=" * 80)
    print(
        f"  {'Strategy':>20} | {'Avg Acc':>8} | {'Gap':>6} | "
        f"{'Min Dom':>8} | {'Max Dom':>8} | {'Total Rank':>10}"
    )
    print("  " + "-" * 75)

    for name, result in all_results.items():
        print(
            f"  {name:>20} | "
            f"{result['avg_accuracy']:8.4f} | "
            f"{result['fairness']['accuracy_gap']:6.4f} | "
            f"{result['fairness']['min_domain_accuracy']:8.4f} | "
            f"{result['fairness']['max_domain_accuracy']:8.4f} | "
            f"{result['total_rank']:>10}"
        )

    print("=" * 80)

    # Plots
    plot_training_curves(
        [
            all_histories[name]
            for name in strategies
        ],
        list(strategies.keys()),
        save_path=os.path.join(
            results_dir,
            "allocation_convergence.png",
        ),
    )

    comparison_data = {}

    for name, result in all_results.items():
        comparison_data[name] = {
            "avg_accuracy": result["avg_accuracy"],
            "per_domain": {
                int(k): v
                for k, v in result["per_domain"].items()
            },
        }

    plot_allocation_comparison(
        comparison_data,
        save_path=os.path.join(
            results_dir,
            "allocation_per_domain.png",
        ),
    )

    # Save results
    output_path = os.path.join(
        results_dir,
        "allocation_comparison.json",
    )

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")
    print("\nExperiment 04 complete!")


if __name__ == "__main__":
    main()
