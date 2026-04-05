"""
Experiment 02: Oracle Rank Search (Brute-Force Grid Search)

Purpose:
    For each client, train LoRA independently at each candidate rank
    {4, 8, 16, 32, 64} and find the actual optimal rank. This produces
    the ground truth for validating the complexity metric.

Outputs:
    - results/oracle_results.json (full results)
    - results/oracle_heatmap.png
    - Console summary of best rank per client

Runtime: ~2 hours (75 training runs, each ~1.5 min)
Note: Resumable — checks for partial results and skips completed runs.
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cifar100_domains import create_federated_datasets
from src.models.lora_resnet import create_lora_resnet, reset_lora_params
from src.utils.visualization import plot_oracle_heatmap


def train_and_evaluate(model, train_loader, test_loader, epochs, lr, device):
    """
    Train LoRA locally and return test accuracy.

    This is a standalone local training (no federation) to isolate
    the question: "what rank does this data distribution need?"
    """
    import torch.nn as nn

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )

    # Train
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x).argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    return correct / total if total > 0 else 0.0


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

    candidate_ranks = config['oracle']['candidate_ranks']
    training_epochs = config['oracle']['training_epochs']
    lr = config['oracle']['learning_rate']

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

    # Load partial results if they exist (resumability)
    partial_path = os.path.join(results_dir, 'oracle_partial.json')
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            oracle_results = json.load(f)
        print(f"Loaded {sum(len(v) for v in oracle_results.values())} partial results")
    else:
        oracle_results = {}

    # Grid search
    n_clients = len(clients_data)
    total_runs = n_clients * len(candidate_ranks)
    completed = sum(len(v) for v in oracle_results.values())

    print(f"\nOracle rank search: {n_clients} clients x {len(candidate_ranks)} ranks = {total_runs} runs")
    print(f"Already completed: {completed} / {total_runs}")

    pbar = tqdm(total=total_runs - completed, desc="Oracle Search")

    for client_id in sorted(clients_data.keys()):
        cid_str = str(client_id)
        if cid_str not in oracle_results:
            oracle_results[cid_str] = {}

        client_info = clients_data[client_id]

        for rank in candidate_ranks:
            rank_str = str(rank)
            if rank_str in oracle_results[cid_str]:
                continue  # Skip completed

            # Fresh model with this rank
            torch.manual_seed(seed)  # Same init for fair comparison
            model = create_lora_resnet(
                num_classes=config['model']['num_classes'],
                rank=rank,
                alpha=config['model']['lora_alpha'],
                pretrained=True,
                device=device,
            )

            # Train and evaluate
            accuracy = train_and_evaluate(
                model,
                client_info['train_loader'],
                client_info['test_loader'],
                epochs=training_epochs,
                lr=lr,
                device=device,
            )

            oracle_results[cid_str][rank_str] = accuracy
            pbar.set_postfix({
                'client': client_id,
                'rank': rank,
                'acc': f'{accuracy:.4f}',
                'domain': client_info['domain_name'][:15],
            })
            pbar.update(1)

            # Save checkpoint
            with open(partial_path, 'w') as f:
                json.dump(oracle_results, f, indent=2)

            # Free memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pbar.close()

    # Save final results
    output_path = os.path.join(results_dir, 'oracle_results.json')
    with open(output_path, 'w') as f:
        json.dump(oracle_results, f, indent=2)
    print(f"\nOracle results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print(f"{'Client':>7} | {'Domain':>30} | ", end="")
    for r in candidate_ranks:
        print(f"{'r=' + str(r):>8} | ", end="")
    print(f"{'Best':>6}")
    print("-" * 70)

    for cid in sorted(clients_data.keys()):
        cid_str = str(cid)
        info = clients_data[cid]
        best_rank = max(oracle_results[cid_str], key=oracle_results[cid_str].get)
        best_acc = oracle_results[cid_str][best_rank]

        print(f"  {cid:5d} | {info['domain_name']:>30s} | ", end="")
        for r in candidate_ranks:
            acc = oracle_results[cid_str].get(str(r), 0)
            marker = " *" if str(r) == best_rank else "  "
            print(f"{acc:6.4f}{marker} | ", end="")
        print(f"r={best_rank}")

    print("=" * 70)

    # Plot heatmap
    plot_oracle_heatmap(
        oracle_results,
        save_path=os.path.join(results_dir, 'oracle_heatmap.png')
    )

    # Clean up partial file
    if os.path.exists(partial_path):
        os.remove(partial_path)

    print("\nExperiment 02 complete!")


if __name__ == '__main__':
    main()
