"""Training-only one-step test of Adam-before versus Adam-after aggregation.

Each batch restarts from the same initial rank-16 factors and zero moments.
It is an arithmetic diagnostic, not a training trajectory or peer deployment.
No test data are loaded, no SVD is applied, and effective updates are averaged.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.feature_cache import tensor_digest, write_json
from experiments.protocol_benchmark import initial_parameters
from src.data.cifar100_domains import get_domain_classes, partition_domain_data_dirichlet
from src.models.lora_resnet import LoRALinear


def gradient(model):
    return torch.cat([model.lora_A.grad.detach().reshape(-1), model.lora_B.grad.detach().reshape(-1)])


def effective_delta(model, alpha):
    return (alpha / model.lora_A.shape[0]) * (model.lora_B.detach() @ model.lora_A.detach())


def vector_comparison(first, second):
    first, second = first.reshape(-1), second.reshape(-1)
    norm_first, norm_second = float(first.norm()), float(second.norm())
    cosine = float(torch.dot(first, second) / (first.norm() * second.norm())) if norm_first and norm_second else None
    return {"global_norm": norm_first, "averaged_local_norm": norm_second,
            "cosine": cosine,
            "relative_difference_to_global": float((first - second).norm()) / norm_first if norm_first else None,
            "max_abs_difference": float((first - second).abs().max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ownership-record-dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.batches < 1 or len(args.seeds) != len(set(args.seeds)):
        parser.error("require positive batch count and unique seeds")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    train = torch.load(args.cache_dir / "train.pt", map_location="cpu", weights_only=True)
    cache_manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    assert tensor_digest(train["features"], train["labels"]) == cache_manifest["train"]["sha256"]
    train_labels = train["labels"].numpy()
    features = train["features"].to(device=args.device, dtype=torch.float64)
    labels = train["labels"].to(args.device)
    alpha, rank, batch_size, lr, weight_decay = 32., 16, 128, .001, .0001
    results = {"status": "running", "host": __import__("socket").gethostname(),
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "training_data_sha256": cache_manifest["train"]["sha256"],
               "test_data_accessed": False, "dtype": "float64", "rank": rank, "alpha": alpha,
               "optimizer": {"name": "Adam", "lr": lr, "weight_decay": weight_decay,
                             "fresh_moments_for_every_batch_and_arm": True},
               "scope": "one-step arithmetic diagnostic; no training-trajectory or convergence claim",
               "protocol": "original non-IID ownership, same initialization for each batch, sample-count-weighted effective-update average; no SVD",
               "gradient_tolerance": {"atol": 2e-11, "rtol": 2e-10}, "batches": []}
    output_path = args.output / "results.json"
    write_json(output_path, results)
    try:
        for seed in args.seeds:
            owners = torch.full((len(labels),), -1, dtype=torch.long, device=args.device)
            for domain in range(5):
                indices = np.flatnonzero(np.isin(train_labels, get_domain_classes(domain)))
                parts = partition_domain_data_dirichlet(indices, train_labels, 3, alpha=.5, seed=seed + domain)
                for local_id, part in enumerate(parts):
                    index = torch.tensor(part, dtype=torch.long, device=args.device)
                    assert torch.all(owners[index] == -1)
                    owners[index] = 3 * domain + local_id
            assert torch.all(owners >= 0)
            owner_hash = tensor_digest(owners)
            if args.ownership_record_dir:
                reference = json.loads((args.ownership_record_dir / f"seed{seed}.json").read_text())
                assert owner_hash == reference["ownership_sha256"]
            initial = initial_parameters(features.shape[1], rank, seed)
            model = LoRALinear(nn.Linear(features.shape[1], 100), rank, alpha).to(device=args.device, dtype=torch.float64)
            with torch.no_grad():
                model.linear.weight.copy_(initial["weight"])
                model.linear.bias.copy_(initial["bias"])
                model.lora_A.copy_(initial["A"])
                model.lora_B.zero_()
            initial_hash = tensor_digest(*model.state_dict().values())
            rng = torch.Generator().manual_seed(seed + 1009)
            permutation = torch.randperm(len(labels), generator=rng).to(args.device)
            for batch_number in range(args.batches):
                batch = permutation[batch_number * batch_size:(batch_number + 1) * batch_size]
                if len(batch) != batch_size:
                    raise ValueError("requested diagnostic batches exceed training epoch")
                x, y = features[batch], labels[batch]
                global_model = copy.deepcopy(model)
                global_optimizer = torch.optim.Adam([global_model.lora_A, global_model.lora_B], lr=lr, weight_decay=weight_decay)
                global_loss = nn.functional.cross_entropy(global_model(x), y)
                global_loss.backward()
                pooled_gradient = gradient(global_model)
                global_optimizer.step()
                pooled_delta = effective_delta(global_model, alpha)
                local_delta = torch.zeros_like(pooled_delta)
                weighted_gradient = torch.zeros_like(pooled_gradient)
                counts = []
                for cid in range(15):
                    index = batch[owners[batch] == cid]
                    counts.append(len(index))
                    if not len(index):
                        continue
                    client_model = copy.deepcopy(model)
                    optimizer = torch.optim.Adam([client_model.lora_A, client_model.lora_B], lr=lr, weight_decay=weight_decay)
                    local_loss = nn.functional.cross_entropy(client_model(features[index]), labels[index])
                    local_loss.backward()
                    fraction = len(index) / len(batch)
                    weighted_gradient.add_(gradient(client_model), alpha=fraction)
                    optimizer.step()
                    local_delta.add_(effective_delta(client_model, alpha), alpha=fraction)
                torch.testing.assert_close(weighted_gradient, pooled_gradient, atol=2e-11, rtol=2e-10)
                with torch.no_grad():
                    global_after = nn.functional.cross_entropy(nn.functional.linear(x, model.linear.weight + pooled_delta, model.linear.bias), y)
                    local_after = nn.functional.cross_entropy(nn.functional.linear(x, model.linear.weight + local_delta, model.linear.bias), y)
                row = {"seed": seed, "batch": batch_number + 1, "n_examples": len(batch),
                       "client_sample_counts": counts, "ownership_sha256": owner_hash,
                       "initial_state_sha256": initial_hash,
                       "gradient_comparison": vector_comparison(pooled_gradient, weighted_gradient),
                       "effective_update_comparison": vector_comparison(pooled_delta, local_delta),
                       "initial_train_batch_loss": float(global_loss.detach()),
                       "global_adam_train_batch_loss": float(global_after),
                       "averaged_local_adam_train_batch_loss": float(local_after),
                       "global_adam_loss_change": float(global_after - global_loss.detach()),
                       "averaged_local_adam_loss_change": float(local_after - global_loss.detach())}
                results["batches"].append(row)
                write_json(output_path, results)
        results["status"] = "completed"
        for metric in ("cosine", "relative_difference_to_global", "global_norm", "averaged_local_norm"):
            values = [row["effective_update_comparison"][metric] for row in results["batches"]]
            results[f"effective_update_{metric}"] = {"mean": float(np.mean(values)), "min": min(values), "max": max(values)}
        for metric in ("global_adam_loss_change", "averaged_local_adam_loss_change"):
            values = [row[metric] for row in results["batches"]]
            results[metric] = {"mean": float(np.mean(values)), "min": min(values), "max": max(values)}
        results["max_gradient_abs_error"] = max(row["gradient_comparison"]["max_abs_difference"] for row in results["batches"])
        write_json(output_path, results)
        print(json.dumps({key: value for key, value in results.items() if key != "batches"}, indent=2))
    except BaseException as error:
        results.update({"status": "failed", "failure_type": type(error).__name__, "failure": str(error)})
        write_json(output_path, results)
        raise


if __name__ == "__main__":
    main()
