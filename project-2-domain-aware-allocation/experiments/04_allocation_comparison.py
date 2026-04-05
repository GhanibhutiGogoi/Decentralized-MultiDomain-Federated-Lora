"""
Experiment 04: Allocation Strategy Comparison

Purpose:
    Compare three LoRA rank allocation strategies in federated learning:
    1. Uniform: all clients get rank 16
    2. Random: each client gets a random rank from {4, 8, 16, 32, 64}
    3. Domain-Aware: rank allocated based on domain complexity score

    Uses HeteroFedAvg to handle mixed LoRA ranks across clients.

Outputs:
    - results/allocation_comparison.json
    - results/allocation_convergence.png
    - results/allocation_per_domain.png
    - Console comparison table

Runtime: ~30 minutes (3 runs of 50 rounds)
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cifar100_domains import create_federated_datasets, print_data_summary
from src.models.lora_resnet import create_lora_resnet
from src.federated.client import FederatedClient
from src.federated.hetero_fedavg import HeteroFedAvgServer
from src.allocation.rank_allocator import RankAllocator
from src.complexity.domain_complexity import DomainComplexityAnalyzer
from src.utils.metrics import compute_fairness_metrics
from src.utils.visualization import (
    plot_training_curves,
    plot_allocation_comparison,
)


def run_fedavg_with_allocation(clients_data, rank_assignments, config, device, desc=""):
    """
    Create clients with specified rank assignments and run HeteroFedAvg.

    Args:
        clients_data: dict from create_federated_datasets
        rank_assignments: dict mapping client_id -> rank
        config: config dict
        device: compute device
        desc: description for progress bar

    Returns:
        history dict from HeteroFedAvgServer.run()
    """
    clients = []
    for cid in sorted(clients_data.keys()):
        rank = rank_assignments[cid]
        model = create_lora_resnet(
            num_classes=config['model']['num_classes'],
            rank=rank,
            alpha=config['model']['lora_alpha'],
            pretrained=True,
            device=device,
        )
        client = FederatedClient(
            client_id=cid,
            model=model,
            train_loader=clients_data[cid]['train_loader'],
            test_loader=clients_data[cid]['test_loader'],
            domain_id=clients_data[cid]['domain_id'],
            lr=config['training']['learning_rate'],
            local_epochs=config['training']['local_epochs'],
            device=device,
        )
        clients.append(client)

    server = HeteroFedAvgServer(
        clients=clients,
        rank_assignments=rank_assignments,
        alpha=config['model']['lora_alpha'],
        device=device,
    )

    history = server.run(
        n_rounds=config['training']['n_rounds'],
        verbose=True,
    )

    return history


def main():
    # Load config
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'configs', 'default_config.yaml'
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = config['training']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = 'cuda' if torch.cuda.is_available() else (
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
    print(f"Using device: {device}")

    results_dir = config['logging']['results_dir']
    os.makedirs(results_dir, exist_ok=True)

    # Create datasets
    print("\nCreating federated datasets...")
    clients_data, train_dataset, test_dataset = create_federated_datasets(
        data_dir=config['data']['data_dir'],
        n_domains=config['data']['n_domains'],
        clients_per_domain=config['data']['clients_per_domain'],
        dirichlet_alpha=config['data']['dirichlet_alpha'],
        batch_size=config['data']['batch_size'],
        seed=seed,
    )
    n_clients = len(clients_data)

    # === Get complexity scores ===
    # Try loading from experiment 01; if not available, compute fresh
    complexity_path = os.path.join(results_dir, 'complexity_scores.json')
    if os.path.exists(complexity_path):
        print("\nLoading complexity scores from experiment 01...")
        with open(complexity_path) as f:
            complexity_data = json.load(f)
        complexity_scores = {int(k): v['score'] for k, v in complexity_data.items()}
    else:
        print("\nComputing complexity scores (experiment 01 not found)...")
        analyzer = DomainComplexityAnalyzer(
            device=device,
            n_feature_samples=config['complexity']['n_feature_samples'],
        )
        complexity_results = analyzer.analyze_all_clients(clients_data, train_dataset)
        complexity_scores = {cid: r['score'] for cid, r in complexity_results.items()}

    # === Define allocation strategies ===
    strategies = {
        'Uniform (r=16)': RankAllocator.uniform(n_clients, rank=16),
        'Random': RankAllocator.random(n_clients, seed=seed),
        'Domain-Aware': RankAllocator.domain_aware(
            complexity_scores,
            min_rank=config['allocation']['min_rank'],
            max_rank=config['allocation']['max_rank'],
            base_rank=config['allocation']['base_rank'],
        ),
    }

    # Print allocations
    print("\n" + "=" * 60)
    print("Rank Allocations:")
    print("=" * 60)
    for name, alloc in strategies.items():
        ranks = list(alloc.values())
        print(f"\n  {name}:")
        print(f"    Ranks: {ranks}")
        print(f"    Mean: {np.mean(ranks):.1f}, Total params: proportional to sum={sum(ranks)}")
    print("=" * 60)

    # === Run experiments ===
    all_histories = {}
    all_results = {}

    for name, rank_assignments in strategies.items():
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        history = run_fedavg_with_allocation(
            clients_data, rank_assignments, config, device, desc=name
        )
        all_histories[name] = history

        # Final metrics
        final_acc = history['avg_accuracy'][-1]
        final_domain = history['per_domain_accuracy'][-1]
        fairness = compute_fairness_metrics(final_domain)

        all_results[name] = {
            'avg_accuracy': final_acc,
            'per_domain': {str(k): v for k, v in final_domain.items()},
            'fairness': fairness,
            'rank_assignments': {str(k): v for k, v in rank_assignments.items()},
            'total_rank': sum(rank_assignments.values()),
        }

        print(f"\n  Final accuracy: {final_acc:.4f}")
        print(f"  Fairness gap: {fairness['accuracy_gap']:.4f}")

    # === Comparison Summary ===
    print("\n" + "=" * 80)
    print("ALLOCATION STRATEGY COMPARISON")
    print("=" * 80)
    print(f"  {'Strategy':>20} | {'Avg Acc':>8} | {'Gap':>6} | {'Min Dom':>8} | {'Max Dom':>8} | {'Total Rank':>10}")
    print("  " + "-" * 75)

    for name, res in all_results.items():
        print(f"  {name:>20} | {res['avg_accuracy']:8.4f} | "
              f"{res['fairness']['accuracy_gap']:6.4f} | "
              f"{res['fairness']['min_domain_accuracy']:8.4f} | "
              f"{res['fairness']['max_domain_accuracy']:8.4f} | "
              f"{res['total_rank']:>10}")
    print("=" * 80)

    # === Plots ===
    # Convergence curves
    plot_training_curves(
        [all_histories[name] for name in strategies],
        list(strategies.keys()),
        save_path=os.path.join(results_dir, 'allocation_convergence.png'),
    )

    # Per-domain comparison
    comparison_data = {}
    for name, res in all_results.items():
        comparison_data[name] = {
            'avg_accuracy': res['avg_accuracy'],
            'per_domain': {int(k): v for k, v in res['per_domain'].items()},
        }
    plot_allocation_comparison(
        comparison_data,
        save_path=os.path.join(results_dir, 'allocation_per_domain.png'),
    )

    # === Save results ===
    output_path = os.path.join(results_dir, 'allocation_comparison.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    print("\nExperiment 04 complete!")


if __name__ == '__main__':
    main()
