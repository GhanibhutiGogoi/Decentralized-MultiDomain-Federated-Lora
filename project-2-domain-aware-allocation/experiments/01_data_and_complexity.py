"""
Experiment 01: Data Distribution Analysis & Domain Complexity Metrics

Purpose:
    Create the federated data splits and compute the domain complexity
    score for all 15 clients. This validates the complexity metric
    and provides the foundation for rank allocation.

Outputs:
    - results/complexity_scores.json
    - results/complexity_breakdown.png
    - results/data_distribution.png
    - Console table of all metrics

Runtime: ~5 minutes
"""

import os
import sys
import json
import yaml
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cifar100_domains import create_federated_datasets, print_data_summary
from src.complexity.domain_complexity import DomainComplexityAnalyzer
from src.utils.visualization import plot_complexity_breakdown, plot_data_distribution


def main():
    # Load config
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'configs', 'default_config.yaml'
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Setup
    seed = config['training']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = 'cuda' if torch.cuda.is_available() else (
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
    print(f"Using device: {device}")

    results_dir = config['logging']['results_dir']
    os.makedirs(results_dir, exist_ok=True)

    # Step 1: Create federated datasets
    print("\n[Step 1] Creating federated datasets...")
    clients_data, train_dataset, test_dataset = create_federated_datasets(
        data_dir=config['data']['data_dir'],
        n_domains=config['data']['n_domains'],
        clients_per_domain=config['data']['clients_per_domain'],
        dirichlet_alpha=config['data']['dirichlet_alpha'],
        batch_size=config['data']['batch_size'],
        seed=seed,
    )
    print_data_summary(clients_data)

    # Plot data distribution
    plot_data_distribution(
        clients_data,
        save_path=os.path.join(results_dir, 'data_distribution.png')
    )

    # Step 2: Compute complexity scores
    print("\n[Step 2] Computing domain complexity scores...")
    analyzer = DomainComplexityAnalyzer(
        device=device,
        n_feature_samples=config['complexity']['n_feature_samples'],
    )

    complexity_results = analyzer.analyze_all_clients(clients_data, train_dataset)

    # Step 3: Print summary table
    print("\n" + "=" * 100)
    print(f"{'Client':>7} | {'Domain':>6} | {'Domain Name':>30} | "
          f"{'Samples':>7} | {'Entropy':>8} | {'Diversity':>9} | "
          f"{'IntrDim':>7} | {'Difficulty':>10} | {'Imbalance':>9} | {'SCORE':>6}")
    print("-" * 100)

    for cid in sorted(complexity_results.keys()):
        r = complexity_results[cid]
        print(f"  {cid:5d} | {r['domain_id']:6d} | {r['domain_name']:>30s} | "
              f"{r['n_samples']:7d} | {r['entropy']:8.4f} | {r['diversity']:9.4f} | "
              f"{r['intrinsic_dim']:7.4f} | {r['task_difficulty']:10.4f} | "
              f"{r['data_imbalance']:9.4f} | {r['score']:6.4f}")

    print("=" * 100)

    # Domain-level summary
    print("\nPer-domain average complexity:")
    domain_scores = {}
    for cid, r in complexity_results.items():
        did = r['domain_id']
        if did not in domain_scores:
            domain_scores[did] = []
        domain_scores[did].append(r['score'])

    for did in sorted(domain_scores.keys()):
        scores = domain_scores[did]
        print(f"  Domain {did} ({complexity_results[did * 3]['domain_name']}): "
              f"avg={np.mean(scores):.4f}, std={np.std(scores):.4f}")

    # Step 4: Save results
    # Convert to JSON-serializable format
    save_results = {}
    for cid, r in complexity_results.items():
        save_results[str(cid)] = {k: float(v) if isinstance(v, (float, np.floating)) else v
                                   for k, v in r.items()}

    output_path = os.path.join(results_dir, 'complexity_scores.json')
    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nComplexity scores saved to {output_path}")

    # Step 5: Plot complexity breakdown
    plot_complexity_breakdown(
        complexity_results,
        save_path=os.path.join(results_dir, 'complexity_breakdown.png')
    )

    print("\nExperiment 01 complete!")


if __name__ == '__main__':
    main()
