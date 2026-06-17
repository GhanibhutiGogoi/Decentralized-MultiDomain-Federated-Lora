"""
Experiment 05: Alpha Search with Fixed Oracle Rank

Purpose:
    For each client, use the best rank from oracle_results.json,
    then search over candidate LoRA alpha values.

Outputs:
    - results/experiment_05_alpha_search/alpha_search_results.json
    - results/experiment_05_alpha_search/best_alpha_per_client.json
    - results/experiment_05_alpha_search/alpha_heatmap.png
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cifar100_domains import create_federated_datasets
from src.models.lora_resnet import create_lora_resnet


def train_and_evaluate(model, train_loader, test_loader, epochs, lr, device):
    import torch.nn as nn

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            preds = model(batch_x).argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    return correct / total if total > 0 else 0.0


def get_best_rank_per_client(oracle_results):
    best_ranks = {}

    for client_id, rank_results in oracle_results.items():
        best_rank = max(rank_results, key=rank_results.get)
        best_ranks[client_id] = int(best_rank)

    return best_ranks


def plot_alpha_heatmap(alpha_results, candidate_alphas, save_path):
    client_ids = sorted(alpha_results.keys(), key=lambda x: int(x))

    heatmap = []
    for cid in client_ids:
        row = []
        for alpha in candidate_alphas:
            row.append(alpha_results[cid].get(str(alpha), np.nan))
        heatmap.append(row)

    heatmap = np.array(heatmap)

    plt.figure(figsize=(10, 7))
    plt.imshow(heatmap, aspect="auto")
    plt.colorbar(label="Accuracy")

    plt.xticks(
        ticks=np.arange(len(candidate_alphas)),
        labels=candidate_alphas,
    )
    plt.yticks(
        ticks=np.arange(len(client_ids)),
        labels=client_ids,
    )

    plt.xlabel("LoRA Alpha")
    plt.ylabel("Client ID")
    plt.title("Alpha Search Heatmap (Fixed Oracle Rank)")

    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            value = heatmap[i, j]
            if not np.isnan(value):
                plt.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
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

    results_root = config["logging"]["results_dir"]
    exp5_dir = os.path.join(results_root, "experiment_05_alpha_search")
    os.makedirs(exp5_dir, exist_ok=True)

    oracle_path = os.path.join(
        results_root,
        "experiment_02_oracle_rank_search",
        "oracle_results.json",
    )

    if not os.path.exists(oracle_path):
        oracle_path = os.path.join(results_root, "oracle_results.json")

    if not os.path.exists(oracle_path):
        raise FileNotFoundError(
            "oracle_results.json not found. Expected either "
            "results/experiment_02_oracle_rank_search/oracle_results.json "
            "or results/oracle_results.json"
        )

    with open(oracle_path) as f:
        oracle_results = json.load(f)

    best_ranks = get_best_rank_per_client(oracle_results)

    candidate_alphas = [4, 8, 16, 32, 64, 128]
    training_epochs = config["oracle"]["training_epochs"]
    lr = config["oracle"]["learning_rate"]

    print("\nCreating federated datasets...")
    clients_data, train_dataset, test_dataset = create_federated_datasets(
        data_dir=config["data"]["data_dir"],
        n_domains=config["data"]["n_domains"],
        clients_per_domain=config["data"]["clients_per_domain"],
        dirichlet_alpha=config["data"]["dirichlet_alpha"],
        batch_size=config["data"]["batch_size"],
        seed=seed,
    )

    alpha_results = {}
    best_alpha_per_client = {}

    total_runs = len(clients_data) * len(candidate_alphas)
    pbar = tqdm(total=total_runs, desc="Alpha Search")

    for client_id in sorted(clients_data.keys()):
        cid_str = str(client_id)
        alpha_results[cid_str] = {}

        fixed_rank = best_ranks[cid_str]
        client_info = clients_data[client_id]

        for alpha in candidate_alphas:
            torch.manual_seed(seed)

            model = create_lora_resnet(
                num_classes=config["model"]["num_classes"],
                rank=fixed_rank,
                alpha=alpha,
                pretrained=True,
                device=device,
            )

            accuracy = train_and_evaluate(
                model=model,
                train_loader=client_info["train_loader"],
                test_loader=client_info["test_loader"],
                epochs=training_epochs,
                lr=lr,
                device=device,
            )

            alpha_results[cid_str][str(alpha)] = accuracy

            pbar.set_postfix({
                "client": client_id,
                "rank": fixed_rank,
                "alpha": alpha,
                "acc": f"{accuracy:.4f}",
            })
            pbar.update(1)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        best_alpha = max(alpha_results[cid_str], key=alpha_results[cid_str].get)

        best_alpha_per_client[cid_str] = {
            "fixed_rank": fixed_rank,
            "best_alpha": int(best_alpha),
            "best_accuracy": alpha_results[cid_str][best_alpha],
            "all_alpha_results": alpha_results[cid_str],
        }

    pbar.close()

    alpha_results_path = os.path.join(exp5_dir, "alpha_search_results.json")
    best_alpha_path = os.path.join(exp5_dir, "best_alpha_per_client.json")
    heatmap_path = os.path.join(exp5_dir, "alpha_heatmap.png")

    with open(alpha_results_path, "w") as f:
        json.dump(alpha_results, f, indent=2)

    with open(best_alpha_path, "w") as f:
        json.dump(best_alpha_per_client, f, indent=2)

    plot_alpha_heatmap(alpha_results, candidate_alphas, heatmap_path)

    print("\nExperiment 05 complete!")
    print(f"Saved: {alpha_results_path}")
    print(f"Saved: {best_alpha_path}")
    print(f"Saved: {heatmap_path}")


if __name__ == "__main__":
    main()
