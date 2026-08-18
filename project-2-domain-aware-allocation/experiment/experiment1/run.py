"""Project 2 Experiment 1: domain signals and marginal contribution.

This runner reuses Project 1's implemented mathematics directly. It only
changes client data partitioning and adds measurement/analysis hooks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT2_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_CODE_ROOT = PROJECT2_ROOT / "experiment"
for path in (PROJECT2_ROOT, EXPERIMENT_CODE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from experiment1.project1_bridge import activate_project1_imports  # noqa: E402

activate_project1_imports()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from config import (  # noqa: E402
    BATCH_TO_MAX_RANK,
    CLIENT_BATCH_SIZES,
    CLIENT_EPOCHS,
    FIXED_RANK,
    LORA_A_SUFFIXES,
    LORA_B_SUFFIXES,
    LORA_SUFFIXES,
    NUM_CLIENTS,
    NUM_ROUNDS,
)
from Federated.client import compute_quality_score, train_client  # noqa: E402
from Federated.fedavg_aggregation import fedavg_quality_weighted  # noqa: E402
from Federated.utilities import evaluate  # noqa: E402
from rank_allocation.LoRa_rank_projection import load_global_state  # noqa: E402
from rank_allocation.rank_selector import estimate_optimal_rank  # noqa: E402
from Source.Models import AudioCNN, CNN, LSTMModel, MLP, TabularMLP  # noqa: E402
from Source.datasets.audio import get_audio  # noqa: E402
from Source.datasets.image import get_cifar10, get_fashion_mnist  # noqa: E402
from Source.datasets.tabular import get_tabular  # noqa: E402
from Source.datasets.text import AGNewsDataset, get_agnews  # noqa: E402

from experiment1.analysis import run_statistical_analysis  # noqa: E402
from experiment1.contribution import (  # noqa: E402
    evaluate_leave_one_client_out,
    flatten_update_vector,
)
from experiment1.partitioning import PartitionConfig, make_client_loaders  # noqa: E402
from experiment1.signals import (  # noqa: E402
    label_distribution_records,
    save_label_distribution_outputs,
    update_dissimilarity_records,
)
from experiment1.visualization import (  # noqa: E402
    plot_client_histograms,
    plot_label_heatmap,
    plot_sample_counts,
    plot_signal_vs_contribution,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_ROOT = PROJECT2_ROOT / "outputs"
EXPERIMENT_NAME = "exp1"
OUTPUT_DIR = OUTPUT_ROOT / EXPERIMENT_NAME


def load_experiments():
    """Load the exact five Project 1 benchmark tasks."""
    print("\n=== Loading Project 1 Benchmark Suite ===")
    cifar_train, cifar_test, cifar_testloader = get_cifar10()
    fashion_train, fashion_test, fashion_testloader = get_fashion_mnist()
    agnews_train, agnews_test, agnews_testloader = get_agnews()
    tabular_train, tabular_test, tabular_testloader = get_tabular()
    audio_train, audio_test, audio_testloader = get_audio()

    print(f"  CIFAR-10  train={len(cifar_train)}, test={len(cifar_test)}")
    print(f"  Fashion   train={len(fashion_train)}, test={len(fashion_test)}")
    print(f"  AG News   train={len(agnews_train)}, test={len(agnews_test)}")
    print(f"  Tabular   train={len(tabular_train)}, test={len(tabular_test)}")
    print(f"  Audio     train={len(audio_train)}, test={len(audio_test)}")

    vocab = AGNewsDataset.VOCAB_SIZE
    return [
        ("CIFAR-CNN", lambda r: CNN(3, 10, r), cifar_train, cifar_testloader),
        ("Fashion-MLP", lambda r: MLP(28 * 28, 10, r), fashion_train, fashion_testloader),
        (
            "AGNews-LSTM",
            lambda r: LSTMModel(vocab, 64, 128, 2, 4, r),
            agnews_train,
            agnews_testloader,
        ),
        (
            "Tabular-MLP",
            lambda r: TabularMLP(tabular_train.in_dim, tabular_train.num_classes, r),
            tabular_train,
            tabular_testloader,
        ),
        (
            "Audio-1DCNN",
            lambda r: AudioCNN(1, audio_train.NUM_CLASSES, r),
            audio_train,
            audio_testloader,
        ),
    ]


def _records_by_client(records):
    return {record["client_id"]: record for record in records}


def run_task(
    task_name,
    model_fn,
    trainset,
    testloader,
    partition_config: PartitionConfig,
    output_dir: Path,
    num_rounds: int,
):
    """Run adaptive Project 1 training with Experiment 1 measurements."""
    print(f"\n{'=' * 70}\n  EXPERIMENT 1 TASK: {task_name}\n{'=' * 70}")
    print(
        f"  partition={partition_config.strategy} "
        f"alpha={partition_config.alpha} seed={partition_config.seed}"
    )

    loaders, metadata = make_client_loaders(
        trainset,
        CLIENT_BATCH_SIZES,
        partition_config,
    )

    label_records, label_payload = label_distribution_records(
        task_name=task_name,
        labels=metadata["labels"],
        indices_by_client=metadata["indices_by_client"],
        num_classes=metadata["num_classes"],
    )
    label_by_client = _records_by_client(label_records)

    figure_dir = output_dir / "figures"
    plot_label_heatmap(task_name, label_payload["client_class_frequency"], figure_dir)
    plot_client_histograms(task_name, label_payload["client_class_counts"], figure_dir)
    plot_sample_counts(task_name, label_payload["client_class_counts"], figure_dir)

    loss_fn = nn.CrossEntropyLoss()
    global_state = model_fn(FIXED_RANK).to(DEVICE).state_dict()
    round_rows = []
    accuracy_curve = []

    for round_id in range(1, num_rounds + 1):
        weights = []
        samples = []
        quality_scores = []
        local_losses = []
        ranks = []
        update_vectors = []

        for client_id, loader in enumerate(loaders):
            batch_size = CLIENT_BATCH_SIZES[client_id]
            probe_rank = BATCH_TO_MAX_RANK[batch_size]
            probe = model_fn(probe_rank).to(DEVICE)
            load_global_state(probe, global_state)

            chosen_rank = estimate_optimal_rank(probe, loader, loss_fn, batch_size)
            ranks.append(chosen_rank)

            local = model_fn(chosen_rank).to(DEVICE)
            load_global_state(local, global_state)

            state, sample_count = train_client(local, loader, CLIENT_EPOCHS, DEVICE)
            quality = compute_quality_score(local, loader, loss_fn, DEVICE)
            local_loss = (1.0 / quality) - 1.0

            weights.append(state)
            samples.append(sample_count)
            quality_scores.append(quality)
            local_losses.append(local_loss)
            update_vectors.append(
                flatten_update_vector(
                    state,
                    global_state,
                    LORA_A_SUFFIXES,
                    LORA_B_SUFFIXES,
                    LORA_SUFFIXES,
                )
            )

        update_records = update_dissimilarity_records(
            task_name,
            round_id,
            update_vectors,
        )
        update_by_client = _records_by_client(update_records)

        ref_sd = model_fn(FIXED_RANK).to(DEVICE).state_dict()
        global_state, loo_rows = evaluate_leave_one_client_out(
            weights=weights,
            samples=samples,
            quality_scores=quality_scores,
            target_rank=FIXED_RANK,
            ref_sd=ref_sd,
            model_fn=model_fn,
            testloader=testloader,
            device=DEVICE,
            aggregate_fn=fedavg_quality_weighted,
            evaluate_fn=evaluate,
            load_global_state_fn=load_global_state,
        )

        full_accuracy = loo_rows[0]["full_accuracy"] if loo_rows else 0.0
        accuracy_curve.append(full_accuracy)

        for loo_row in loo_rows:
            client_id = loo_row["client_id"]
            label_row = label_by_client[client_id]
            update_row = update_by_client[client_id]
            round_rows.append(
                {
                    "task": task_name,
                    "round": round_id,
                    "client_id": client_id,
                    "partition_strategy": partition_config.strategy,
                    "partition_alpha": partition_config.alpha,
                    "partition_seed": partition_config.seed,
                    "hardware_batch_size": CLIENT_BATCH_SIZES[client_id],
                    "train_samples_seen": int(samples[client_id]),
                    "partition_samples": int(label_row["num_samples"]),
                    "adaptive_rank": int(ranks[client_id]),
                    "local_loss": float(local_losses[client_id]),
                    "quality_score": float(quality_scores[client_id]),
                    "entropy": float(label_row["entropy"]),
                    "normalized_entropy": float(label_row["normalized_entropy"]),
                    "class_imbalance_ratio": float(label_row["class_imbalance_ratio"]),
                    "kl_to_global": float(label_row["kl_to_global"]),
                    "js_to_global": float(label_row["js_to_global"]),
                    "zero_class_count": int(label_row["zero_class_count"]),
                    "update_cosine_distance_to_mean": float(
                        update_row["update_cosine_distance_to_mean"]
                    ),
                    "update_l2_distance_to_mean": float(
                        update_row["update_l2_distance_to_mean"]
                    ),
                    "update_norm": float(update_row["update_norm"]),
                    "full_accuracy": float(loo_row["full_accuracy"]),
                    "loo_accuracy": float(loo_row["loo_accuracy"]),
                    "delta_accuracy": float(loo_row["delta_accuracy"]),
                }
            )

        detail = " | ".join(
            f"C{idx}(r={ranks[idx]},q={quality_scores[idx]:.3f},"
            f"dacc={loo_rows[idx]['delta_accuracy']:.3f})"
            for idx in range(NUM_CLIENTS)
        )
        print(
            f"  Round {round_id}/{num_rounds} | acc={full_accuracy:.2f}% | {detail}"
        )

    return {
        "task": task_name,
        "label_records": label_records,
        "label_payload": label_payload,
        "round_rows": round_rows,
        "accuracy_curve": accuracy_curve,
    }


def save_accuracy_curves(task_results, output_dir: Path):
    rows = []
    for result in task_results:
        for idx, accuracy in enumerate(result["accuracy_curve"], start=1):
            rows.append(
                {
                    "task": result["task"],
                    "round": idx,
                    "full_accuracy": accuracy,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "accuracy_curves.csv", index=False)
    return df


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition",
        default="dirichlet",
        choices=["iid", "legacy_iid", "project1_iid", "dirichlet"],
        help="Client partition strategy. IID variants preserve Project 1 splitting.",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional subset of task names to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for Experiment 1 outputs.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(PROJECT2_ROOT)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    partition_config = PartitionConfig(
        strategy=args.partition,
        alpha=args.alpha,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    print(f"Using device: {DEVICE}")
    print("Reusing Project 1 adaptive rank and aggregation implementations.")
    print(f"Output directory: {output_dir}")

    experiments = load_experiments()
    if args.tasks:
        selected = set(args.tasks)
        experiments = [item for item in experiments if item[0] in selected]
        missing = selected - {item[0] for item in experiments}
        if missing:
            raise ValueError(f"Unknown task name(s): {sorted(missing)}")

    task_results = []
    label_records_all = []
    label_payloads = {}
    measurement_rows = []

    for task_name, model_fn, trainset, testloader in experiments:
        result = run_task(
            task_name=task_name,
            model_fn=model_fn,
            trainset=trainset,
            testloader=testloader,
            partition_config=partition_config,
            output_dir=output_dir,
            num_rounds=args.num_rounds,
        )
        task_results.append(result)
        label_records_all.extend(result["label_records"])
        label_payloads[task_name] = result["label_payload"]
        measurement_rows.extend(result["round_rows"])

    label_df = save_label_distribution_outputs(
        label_records_all,
        label_payloads,
        output_dir,
    )
    measurements = pd.DataFrame(measurement_rows)
    measurements.to_csv(output_dir / "per_round_client_measurements.csv", index=False)
    save_accuracy_curves(task_results, output_dir)
    corr_df, reg_df = run_statistical_analysis(measurements, output_dir)
    plot_signal_vs_contribution(measurements, output_dir / "figures")

    manifest = {
        "project": "Project 2",
        "experiment": "Experiment 1",
        "project1_reuse": True,
        "partition": partition_config.__dict__,
        "num_rounds": args.num_rounds,
        "num_clients": NUM_CLIENTS,
        "client_batch_sizes": list(CLIENT_BATCH_SIZES),
        "tasks": [result["task"] for result in task_results],
        "outputs": {
            "label_summary_csv": "label_distribution_summary.csv",
            "label_raw_json": "label_distribution_raw.json",
            "per_round_measurements_csv": "per_round_client_measurements.csv",
            "accuracy_curves_csv": "accuracy_curves.csv",
            "correlations_csv": "signal_contribution_correlations.csv",
            "controlled_regression_csv": "controlled_regression.csv",
            "figures_dir": "figures/",
        },
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== Experiment 1 Complete ===")
    print(f"Label rows: {len(label_df)}")
    print(f"Measurement rows: {len(measurements)}")
    print(f"Correlation rows: {len(corr_df)}")
    print(f"Regression rows: {len(reg_df)}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
