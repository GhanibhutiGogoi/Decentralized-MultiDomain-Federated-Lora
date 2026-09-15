"""Direct pooled-data LoRA baseline for the end-to-end comparison."""
import argparse
from pathlib import Path
import numpy as np
import torch

from experiments.feature_cache import load_features
from experiments.protocol_benchmark import FeatureClient, initial_parameters


def run(seed, data_dir, feature_cache, rounds, rank, device):
    config = argparse.Namespace(alpha=32.0, lr=0.001, weight_decay=0.0001,
                               local_epochs=1, batch_size=128, eval_batch_size=2048,
                               current_seed=seed)
    train_cpu, test_cpu, _ = load_features(Path(data_dir), Path(feature_cache), torch.device(device), 224, 256, 2)
    train = {k: v.to(device) for k, v in train_cpu.items()}
    test = {k: v.to(device) for k, v in test_cpu.items()}
    initial = initial_parameters(train["features"].shape[1], rank, seed)
    all_train = np.arange(len(train["labels"]), dtype=np.int64)
    all_test = np.arange(len(test["labels"]), dtype=np.int64)
    split = {"train_indices": all_train.tolist(), "test_indices": all_test.tolist()}
    client = FeatureClient(0, 0, initial, rank, config, train, test, split, torch.device(device))
    curve = []
    for _ in range(int(rounds)):
        client.train()
        curve.append(float(client.evaluate()["accuracy"]))
    return {"seed": seed, "rank": rank, "rounds": rounds,
            "final_full_test_accuracy": curve[-1], "accuracy_curve": curve,
            "n_train": len(all_train), "n_test": len(all_test)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--feature-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.set_num_threads(2)
    results = [run(s, args.data_dir, args.feature_cache, args.rounds, args.rank, args.device) for s in args.seeds]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import json
    args.output.write_text(json.dumps({"protocol": "direct pooled-data LoRA", "results": results}, indent=2))
    for r in results:
        print(f"seed={r['seed']} final_full_test_accuracy={r['final_full_test_accuracy']:.4f}", flush=True)


if __name__ == "__main__":
    main()
