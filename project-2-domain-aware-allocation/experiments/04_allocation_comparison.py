"""
Experiment 04: Allocation Strategy Comparison

Purpose:
    Compare LoRA rank allocation strategies in federated learning:
    1. Uniform: all clients get rank 16
    2. Random: each client gets a random rank
    3. Random Matched-Budget: heterogeneous random ranks with fixed total-rank budget
    4. Domain-Aware: rank allocated based on domain complexity score
    5. Dynamic Allocation: adaptive-rank-style output + alpha/scaling policy
    6. Weighted Assignment: client/domain signals -> weights -> fixed-budget rank allocation

    Uses HeteroFedAvg to handle mixed LoRA ranks across clients.

Outputs:
    - results/experiment_04_allocation_comparison/allocation_comparison.json
    - results/experiment_04_allocation_comparison/allocation_convergence.png
    - results/experiment_04_allocation_comparison/allocation_per_domain.png
    - Console comparison table

Note:
    This experiment is a controlled integration step. The matched-budget
    strategies help separate allocation-policy effects from total-rank capacity.
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
from src.allocation.weighted_assignment_policy import weighted_assignment
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
    Build Gabriel-style heterogeneous rank assignments.

    Gabriel's Project 1 configuration maps:
        batch 16  -> max rank 4
        batch 64  -> max rank 8
        batch 256 -> max rank 16

    This integration uses the same 4/8/16 heterogeneous rank pattern as a
    Project 2 bridge input.

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
    Build dynamic allocation using heterogeneous rank output and alpha policy.

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


def build_random_matched_budget_strategy(
    n_clients,
    target_budget=240,
    seed=42,
):
    """
    Build a heterogeneous random baseline with fixed total-rank budget.

    For 15 clients and target_budget=240, this uses a shuffled rank multiset:

        4 clients  -> rank 4
        4 clients  -> rank 8
        4 clients  -> rank 16
        2 clients  -> rank 32
        1 client   -> rank 64

    Total rank:
        4*4 + 4*8 + 4*16 + 2*32 + 1*64 = 240

    This baseline tests whether heterogeneous rank diversity alone explains
    performance differences under the same total rank budget.
    """
    if n_clients != 15 or target_budget != 240:
        raise ValueError(
            "This first matched-budget baseline is defined for "
            "15 clients and total_rank_budget=240."
        )

    ranks = (
        [4] * 4
        + [8] * 4
        + [16] * 4
        + [32] * 2
        + [64] * 1
    )

    rng = np.random.default_rng(seed)
    rng.shuffle(ranks)

    return {
        cid: int(ranks[cid])
        for cid in range(n_clients)
    }


def load_weighted_client_features(results_root, complexity_scores):
    """
    Load client features for weighted assignment.

    Preferred source:
        results/experiment_08_scaling_analysis/scaling_analysis.json

    This file already contains:
        - complexity_score
        - entropy
        - data_imbalance
        - client/domain metadata

    Fallback:
        use complexity_scores only, with entropy and data_imbalance set to 0.
    """
    scaling_path = os.path.join(
        results_root,
        "experiment_08_scaling_analysis",
        "scaling_analysis.json",
    )

    if os.path.exists(scaling_path):
        print(f"\nLoading weighted-assignment features from: {scaling_path}")

        with open(scaling_path, "r") as f:
            scaling_data = json.load(f)

        client_features = {}

        for row in scaling_data["client_table"]:
            client_id = f"client_{row['client_id']}"

            client_features[client_id] = {
                "client_id": row["client_id"],
                "domain_id": row["domain_id"],
                "domain_name": row["domain_name"],
                "complexity_score": row["complexity_score"],
                "entropy": row["entropy"],
                "data_imbalance": row["data_imbalance"],
            }

        return client_features

    print(
        "\nExperiment 08 scaling_analysis.json was not found. "
        "Falling back to complexity scores only for weighted assignment."
    )

    return {
        f"client_{cid}": {
            "client_id": cid,
            "complexity_score": score,
            "entropy": 0.0,
            "data_imbalance": 0.0,
        }
        for cid, score in complexity_scores.items()
    }


def build_weighted_assignment_strategy(
    results_root,
    complexity_scores,
    total_rank_budget=240,
    alpha=64,
):
    """
    Build Weighted Assignment strategy using the weighted assignment policy.

    The formula maps client/domain signals into normalized weights, allocates
    a fixed total-rank budget, maps continuous ranks to allowed LoRA ranks,
    and computes alpha/scaling.
    """
    client_features = load_weighted_client_features(
        results_root=results_root,
        complexity_scores=complexity_scores,
    )

    weighted_results = weighted_assignment(
        client_features=client_features,
        total_rank_budget=total_rank_budget,
        allowed_ranks=[4, 8, 16, 32, 64],
        min_rank=4,
        alpha=alpha,
        lambda_complexity=1.0,
        lambda_entropy=0.5,
        lambda_imbalance=0.5,
    )

    rank_assignments = {
        int(client_id.replace("client_", "")): allocation["rank"]
        for client_id, allocation in weighted_results["allocations"].items()
    }

    return rank_assignments, weighted_results


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

    # Matched-budget reference
    # Uniform rank 16 over 15 clients gives total_rank_budget = 240.
    matched_total_rank_budget = 240

    # Build dynamic allocation strategy
    dynamic_rank_assignments, dynamic_allocations = build_dynamic_allocation_strategy(
        n_clients=n_clients,
        alpha=dynamic_alpha,
    )

    # Build random matched-budget strategy
    random_matched_rank_assignments = build_random_matched_budget_strategy(
        n_clients=n_clients,
        target_budget=matched_total_rank_budget,
        seed=seed,
    )

    # Build weighted assignment strategy
    weighted_rank_assignments, weighted_assignment_results = build_weighted_assignment_strategy(
        results_root=results_root,
        complexity_scores=complexity_scores,
        total_rank_budget=matched_total_rank_budget,
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
                "matched_total_rank_budget": matched_total_rank_budget,
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
                "description": (
                    "Each client receives a random rank. "
                    "This baseline is not matched to a fixed total-rank budget."
                ),
            },
        },

        "Random Matched-Budget": {
            "rank_assignments": random_matched_rank_assignments,
            "alpha": dynamic_alpha,
            "metadata": {
                "type": "random_matched_budget",
                "description": (
                    "Heterogeneous random rank assignment with fixed "
                    "total-rank budget."
                ),
                "target_total_rank_budget": matched_total_rank_budget,
                "purpose": (
                    "Tests whether heterogeneous rank diversity alone explains "
                    "performance differences under a matched capacity budget."
                ),
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
                "note": (
                    "This strategy is kept as the original domain-aware baseline. "
                    "Its total-rank budget may differ from matched-budget baselines."
                ),
            },
        },

        "Dynamic Allocation": {
            "rank_assignments": dynamic_rank_assignments,
            "alpha": dynamic_alpha,
            "metadata": {
                "type": "dynamic_allocation",
                "description": (
                    "Heterogeneous adaptive-rank-style assignments connected with "
                    "Project 2 alpha/scaling policy."
                ),
                "dynamic_allocations": dynamic_allocations,
            },
        },

        "Weighted Assignment": {
            "rank_assignments": weighted_rank_assignments,
            "alpha": dynamic_alpha,
            "metadata": {
                "type": "weighted_assignment",
                "description": (
                    "Client/domain signals are converted into normalized weights "
                    "and used to allocate a fixed total LoRA rank budget."
                ),
                "target_total_rank_budget": matched_total_rank_budget,
                "weighted_assignment_results": weighted_assignment_results,
                "purpose": (
                    "Tests the weighted assignment formula under a matched "
                    "total-rank budget."
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
    print("\n" + "=" * 95)
    print("ALLOCATION STRATEGY COMPARISON")
    print("=" * 95)
    print(
        f"  {'Strategy':>25} | {'Avg Acc':>8} | {'Gap':>6} | "
        f"{'Min Dom':>8} | {'Max Dom':>8} | {'Total Rank':>10}"
    )
    print("  " + "-" * 90)

    for name, result in all_results.items():
        print(
            f"  {name:>25} | "
            f"{result['avg_accuracy']:8.4f} | "
            f"{result['fairness']['accuracy_gap']:6.4f} | "
            f"{result['fairness']['min_domain_accuracy']:8.4f} | "
            f"{result['fairness']['max_domain_accuracy']:8.4f} | "
            f"{result['total_rank']:>10}"
        )

    print("=" * 95)

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
