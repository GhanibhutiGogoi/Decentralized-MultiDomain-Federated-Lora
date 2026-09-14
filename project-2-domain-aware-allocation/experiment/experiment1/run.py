"""Project 2 Experiment 1: domain signals and marginal contribution.

This runner reuses Project 1's implemented mathematics directly. It only
changes client data partitioning and adds measurement/analysis hooks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import hashlib
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
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

from experiment1.analysis import run_statistical_analysis  # noqa: E402
from experiment1.contribution import (  # noqa: E402
    evaluate_leave_one_client_out,
    flatten_update_vector,
)
from framework.datasets import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DatasetConfig,
    DatasetFactory,
    write_dataset_manifest,
)
from framework.datasets.text import AGNewsDataset  # noqa: E402
from framework.partitioning import PartitionConfig, make_client_loaders  # noqa: E402
from framework.utils import (  # noqa: E402
    ensure_disjoint_directory,
    ensure_not_cleanup_parent,
    environment_manifest,
    prepare_output_directory,
    set_reproducibility_seed,
)
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
EXPERIMENT2_OUTPUT_DIR = OUTPUT_ROOT / "exp2"
TASK_ORDER = [
    "CIFAR-CNN",
    "Fashion-MLP",
    "AGNews-LSTM",
    "Tabular-MLP",
    "Audio-1DCNN",
]
CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_OUTPUT_SCHEMA_VERSION = "p2-exp1-output/v2"
CHECKPOINT_RESUME_BOUNDARY = "task-boundary"
MEASUREMENT_NUMERIC_FIELDS = {
    "round", "client_id", "partition_alpha", "partition_seed", "hardware_batch_size",
    "train_samples_seen", "partition_samples", "adaptive_rank", "local_loss", "quality_score",
    "entropy", "normalized_entropy", "class_imbalance_ratio", "kl_to_global", "js_to_global",
    "zero_class_count", "update_cosine_distance_to_mean", "update_l2_distance_to_mean",
    "update_norm", "full_accuracy", "loo_accuracy", "delta_accuracy",
}


class Experiment1CheckpointError(ValueError):
    """Raised when an Experiment 1 checkpoint is malformed or incompatible."""


def _json_safe(value):
    """Convert experiment values to strict JSON-compatible primitives."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _stable_dataset_provenance(manifest: dict) -> dict:
    volatile = {"generated_at_utc", "cache_status", "download_status"}
    datasets = manifest.get("datasets", {}) if isinstance(manifest, dict) else {}
    return {
        task: {key: value for key, value in record.items() if key not in volatile}
        for task, record in datasets.items()
    }


def _atomic_json_write(path: Path, payload: dict) -> None:
    """Atomically persist JSON so interruption cannot leave a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_dataframe_to_csv(frame: pd.DataFrame, path: Path) -> None:
    """Atomically replace a CSV artifact after successful serialization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_revision() -> dict:
    """Capture source identity used to reject incompatible resumes."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT2_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        commit = "unknown"
    hasher = hashlib.sha256()
    repo_root = PROJECT2_ROOT.parent
    source_roots = [PROJECT2_ROOT / "experiment", PROJECT2_ROOT / "framework", repo_root / "project-1-adaptive-rank"]
    files = sorted(path for root in source_roots if root.exists() for path in root.rglob("*.py"))
    for path in files:
        hasher.update(str(path.relative_to(repo_root)).encode())
        hasher.update(path.read_bytes())
    return {"git_commit": commit, "runtime_sources_sha256": hasher.hexdigest()}


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise Experiment1CheckpointError(f"Checkpoint not found: {path}")

    def reject_duplicate_keys(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise Experiment1CheckpointError(f"duplicate JSON key {key!r}")
            out[key] = value
        return out

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
                parse_constant=lambda value: (_ for _ in ()).throw(Experiment1CheckpointError(f"nonfinite JSON constant {value}")),
        )
    except (OSError, json.JSONDecodeError, Experiment1CheckpointError) as exc:
        raise Experiment1CheckpointError(f"Malformed checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise Experiment1CheckpointError("Unsupported or malformed Experiment 1 checkpoint schema")
    if payload.get("status") not in {"running", "completed"}:
        raise Experiment1CheckpointError("Checkpoint status must be running or completed")
    if not isinstance(payload.get("completed_tasks"), list) or not isinstance(payload.get("task_results"), list):
        raise Experiment1CheckpointError("Checkpoint is missing completed_tasks/task_results")
    if payload.get("output_schema_version") != CHECKPOINT_OUTPUT_SCHEMA_VERSION:
        raise Experiment1CheckpointError("Checkpoint is missing the Experiment 1 output schema identity")
    if payload.get("resume_boundary") != CHECKPOINT_RESUME_BOUNDARY:
        raise Experiment1CheckpointError("Checkpoint resume boundary is unsupported")
    run_identity = payload.get("run_identity")
    if not isinstance(run_identity, dict) or not isinstance(payload.get("run_identity_hash"), str):
        raise Experiment1CheckpointError("Checkpoint is missing run identity")
    if payload["run_identity_hash"] != _stable_hash(run_identity):
        raise Experiment1CheckpointError("Checkpoint run identity hash does not match its payload")
    return payload


def _run_identity(task_names, args, partition_config, dataset_manifest: dict) -> dict:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "output_schema_version": CHECKPOINT_OUTPUT_SCHEMA_VERSION,
        "resume_boundary": CHECKPOINT_RESUME_BOUNDARY,
        "tasks": list(task_names),
        "num_rounds": int(args.num_rounds),
        "num_clients": int(NUM_CLIENTS),
        "seed": int(args.seed),
        "task_seed_policy": "seed + task index",
        "task_seeds": {
            task: int(args.seed) + idx for idx, task in enumerate(task_names)
        },
        "partition": _json_safe(partition_config.__dict__),
        "data_root": str(Path(args.data_root).resolve()),
        "synthetic_datasets": sorted(args.synthetic_datasets),
        "download_datasets": bool(args.download_datasets),
        "client_batch_sizes": list(CLIENT_BATCH_SIZES),
        "client_epochs": int(CLIENT_EPOCHS),
        "training": {
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "local_epochs": int(CLIENT_EPOCHS),
        },
        "lora": {
            "fixed_rank": int(FIXED_RANK),
            "batch_to_max_rank": {str(k): int(v) for k, v in BATCH_TO_MAX_RANK.items()},
        },
        "dataset_provenance": _stable_dataset_provenance(dataset_manifest),
        "source_revision": _source_revision(),
    }


def _validate_checkpoint_identity(checkpoint: dict, *, task_names, args, partition_config, dataset_manifest: dict | None = None):
    expected = {
        "output_schema_version": CHECKPOINT_OUTPUT_SCHEMA_VERSION,
        "resume_boundary": CHECKPOINT_RESUME_BOUNDARY,
        "tasks": list(task_names),
        "num_rounds": int(args.num_rounds),
        "num_clients": int(NUM_CLIENTS),
        "seed": int(args.seed),
        "partition": _json_safe(partition_config.__dict__),
        "data_root": str(Path(args.data_root).resolve()),
        "synthetic_datasets": sorted(args.synthetic_datasets),
        "download_datasets": bool(args.download_datasets),
        "task_seed_policy": "seed + task index",
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(
                f"Checkpoint incompatible for {key}: stored={checkpoint.get(key)!r}, expected={value!r}"
            )
    if dataset_manifest is not None:
        expected_identity = _run_identity(task_names, args, partition_config, dataset_manifest)
        if checkpoint.get("run_identity") != expected_identity:
            raise ValueError("Checkpoint run identity differs from the requested run")
        if checkpoint.get("run_identity_hash") != _stable_hash(expected_identity):
            raise ValueError("Checkpoint run identity hash differs from requested run")
    completed = checkpoint["completed_tasks"]
    if completed != list(task_names)[:len(completed)]:
        raise ValueError("Checkpoint completed tasks must be an ordered prefix of the requested tasks")
    if len(checkpoint["task_results"]) != len(completed):
        raise ValueError("Checkpoint task result count does not match completed tasks")
    for task, result in zip(completed, checkpoint["task_results"]):
        required_types = {
            "task": str,
            "is_synthetic": bool,
            "label_records": list,
            "label_payload": dict,
            "round_rows": list,
            "accuracy_curve": list,
        }
        if not isinstance(result, dict) or any(
            not isinstance(result.get(key), value_type) for key, value_type in required_types.items()
        ):
            raise ValueError(f"Checkpoint has incomplete result data for {task}")
        if result["task"] != task or len(result["accuracy_curve"]) != args.num_rounds:
            raise ValueError(f"Checkpoint has incomplete rounds or incorrect task identity for {task}")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in result["accuracy_curve"]):
            raise ValueError(f"Checkpoint has invalid accuracy values for {task}")
        rows = result["round_rows"]
        expected_rows = args.num_rounds * NUM_CLIENTS
        if len(rows) != expected_rows:
            raise ValueError(f"Checkpoint has incomplete client-round rows for {task}")
        seen = {(row.get("round"), row.get("client_id")) for row in rows if isinstance(row, dict)}
        expected = {(round_id, client_id) for round_id in range(1, args.num_rounds + 1) for client_id in range(NUM_CLIENTS)}
        if seen != expected:
            raise ValueError(f"Checkpoint client-round rows are malformed for {task}")
        for row in rows:
            if row.get("task") != task or row.get("is_synthetic") is not result["is_synthetic"]:
                raise ValueError(f"Checkpoint measurement identity differs for {task}")
            if row.get("partition_strategy") != partition_config.strategy:
                raise ValueError(f"Checkpoint measurement partition differs for {task}")
            if any(isinstance(row.get(key), bool) or not isinstance(row.get(key), (int, float))
                   or not math.isfinite(row[key]) for key in MEASUREMENT_NUMERIC_FIELDS):
                raise ValueError(f"Checkpoint has incomplete or nonfinite measurements for {task}")
        labels = result["label_records"]
        if len(labels) != NUM_CLIENTS or any(not isinstance(row, dict) for row in labels):
            raise ValueError(f"Checkpoint label records are incomplete for {task}")
        if {row.get("client_id") for row in labels} != set(range(NUM_CLIENTS)) or any(row.get("task") != task for row in labels):
            raise ValueError(f"Checkpoint label identities differ for {task}")
        payload = result["label_payload"]
        if payload.get("task") != task or any(not isinstance(payload.get(key), list) for key in
                ("global_class_counts", "global_class_frequency", "client_class_counts", "client_class_frequency")):
            raise ValueError(f"Checkpoint label payload is incomplete for {task}")
    if checkpoint["status"] == "completed" and completed != list(task_names):
        raise ValueError("Completed checkpoint does not contain every requested task")


def _is_synthetic(dataset) -> bool:
    return bool(getattr(dataset, "is_synthetic", False))


def load_experiments(
    task_names: list[str],
    data_root: Path,
    download_datasets: bool,
    num_workers: int,
    pin_memory: bool,
    synthetic_datasets: set[str] | None = None,
):
    """Load the exact five Project 1 benchmark tasks."""
    synthetic_datasets = synthetic_datasets or set()
    unknown = set(task_names) - set(TASK_ORDER)
    if unknown:
        raise ValueError(f"Unknown task name(s): {sorted(unknown)}")

    print("\n=== Loading Project 1 Benchmark Suite ===")
    factory = DatasetFactory(
        DatasetConfig(
            data_root=data_root,
            download=download_datasets,
            synthetic=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    )
    bundles = factory.load_tasks(task_names, synthetic_tasks=synthetic_datasets)

    for task_name in task_names:
        metadata = bundles[task_name].metadata
        print(
            f"  {task_name:12s} train={metadata.train_sample_count}, "
            f"test={metadata.test_sample_count}, classes={metadata.class_count}, "
            f"synthetic={metadata.synthetic}, cache={metadata.cache_status}, "
            f"download={metadata.download_status}"
        )

    vocab = AGNewsDataset.VOCAB_SIZE
    experiments = []
    for task_name in task_names:
        bundle = bundles[task_name]
        if task_name == "CIFAR-CNN":
            item = (
                task_name,
                lambda r, b=bundle: CNN(3, b.metadata.class_count, r),
                bundle.train,
                bundle.test_loader,
            )
        elif task_name == "Fashion-MLP":
            item = (
                task_name,
                lambda r, b=bundle: MLP(28 * 28, b.metadata.class_count, r),
                bundle.train,
                bundle.test_loader,
            )
        elif task_name == "AGNews-LSTM":
            item = (
                task_name,
                lambda r: LSTMModel(vocab, 64, 128, 2, 4, r),
                bundle.train,
                bundle.test_loader,
            )
        elif task_name == "Tabular-MLP":
            item = (
                task_name,
                lambda r, b=bundle: TabularMLP(b.train.in_dim, b.train.num_classes, r),
                bundle.train,
                bundle.test_loader,
            )
        elif task_name == "Audio-1DCNN":
            item = (
                task_name,
                lambda r, b=bundle: AudioCNN(1, b.train.NUM_CLASSES, r),
                bundle.train,
                bundle.test_loader,
            )
        else:
            raise ValueError(f"Unknown task name: {task_name}")
        experiments.append(item)
    return experiments, bundles


def _records_by_client(records):
    return {record["client_id"]: record for record in records}


def _record_progress(output_dir: Path, event: str, **payload):
    """Persist completed work before later tasks or analysis can fail."""
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    with (output_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")


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
    task_started = time.monotonic()
    _record_progress(output_dir, "task_started", task=task_name, num_rounds=num_rounds)

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
        is_synthetic=_is_synthetic(trainset),
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
            client_started = time.monotonic()
            print(f"  Round {round_id}/{num_rounds} | client {client_id} training", flush=True)
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
            _record_progress(
                output_dir,
                "client_completed",
                task=task_name,
                round=round_id,
                client_id=client_id,
                adaptive_rank=int(chosen_rank),
                train_samples_seen=int(sample_count),
                quality_score=float(quality),
                elapsed_seconds=time.monotonic() - client_started,
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
                    "is_synthetic": _is_synthetic(trainset),
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
        _record_progress(
            output_dir,
            "round_completed",
            task=task_name,
            round=round_id,
            full_accuracy=float(full_accuracy),
            measurements=round_rows[-len(loo_rows):],
            task_elapsed_seconds=time.monotonic() - task_started,
        )

    _record_progress(
        output_dir,
        "task_completed",
        task=task_name,
        elapsed_seconds=time.monotonic() - task_started,
        accuracy_curve=accuracy_curve,
    )

    return {
        "task": task_name,
        "is_synthetic": _is_synthetic(trainset),
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
    _atomic_dataframe_to_csv(df, output_dir / "accuracy_curves.csv")
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Clean the Experiment 1 output directory before running. By default "
            "a non-empty output directory is rejected."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint.json in --output-dir.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume from an explicit Experiment 1 output directory or checkpoint.json.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Central dataset cache root.",
    )
    parser.add_argument(
        "--download-datasets",
        action="store_true",
        help="Explicitly download missing real datasets instead of failing.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument(
        "--synthetic-datasets",
        nargs="*",
        default=[],
        choices=["AGNews-LSTM", "Tabular-MLP", "Audio-1DCNN"],
        help="Explicitly use synthetic data for named fallback-capable tasks.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(PROJECT2_ROOT)

    if (args.resume or args.resume_from is not None) and args.overwrite:
        raise ValueError("Resume cannot be combined with --overwrite")
    resume_path = args.resume_from
    if args.resume and resume_path is None:
        resume_path = args.output_dir
    if resume_path is not None:
        resume_path = Path(resume_path)
        if resume_path.name == "checkpoint.json":
            resume_checkpoint_path = resume_path
            args.output_dir = resume_path.parent
        else:
            args.output_dir = resume_path
            resume_checkpoint_path = resume_path / "checkpoint.json"
    else:
        resume_checkpoint_path = None

    ensure_disjoint_directory(
        args.output_dir,
        EXPERIMENT2_OUTPUT_DIR,
        output_label="Experiment 1 output directory",
        protected_label="Experiment 2 output directory",
    )
    ensure_not_cleanup_parent(
        args.output_dir,
        OUTPUT_ROOT,
        output_label="Experiment 1 output directory",
        protected_label="shared Project 2 outputs directory",
    )
    if resume_checkpoint_path is None:
        output_dir = prepare_output_directory(
            args.output_dir,
            overwrite=args.overwrite,
            experiment_name="Experiment 1",
            allowed_cleanup_root=OUTPUT_DIR,
            repository_root=PROJECT2_ROOT.parent,
            project_root=PROJECT2_ROOT,
            shared_outputs_root=OUTPUT_ROOT,
        )
    else:
        output_dir = args.output_dir
        if not output_dir.is_dir():
            raise ValueError(f"Resume output directory does not exist: {output_dir}")
    set_reproducibility_seed(args.seed)

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

    task_names = args.tasks if args.tasks else TASK_ORDER
    checkpoint = None
    if resume_checkpoint_path is not None:
        checkpoint = _load_checkpoint(resume_checkpoint_path)
        _validate_checkpoint_identity(
            checkpoint,
            task_names=task_names,
            args=args,
            partition_config=partition_config,
        )
    _record_progress(
        output_dir,
        "run_started",
        tasks=task_names,
        seed=args.seed,
        num_rounds=args.num_rounds,
        partition=partition_config.__dict__,
    )
    experiments, dataset_bundles = load_experiments(
        task_names=task_names,
        data_root=args.data_root,
        download_datasets=args.download_datasets,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        synthetic_datasets=set(args.synthetic_datasets),
    )
    # Build the candidate manifest outside the output directory so a rejected
    # resume cannot overwrite the provenance of the original run.
    with tempfile.TemporaryDirectory(prefix="p2-exp1-manifest-") as staging_dir:
        dataset_manifest = write_dataset_manifest(
            output_dir=Path(staging_dir),
            experiment_name="Experiment 1",
            bundles=dataset_bundles,
        )

    run_identity = _run_identity(task_names, args, partition_config, dataset_manifest)
    identity = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "output_schema_version": CHECKPOINT_OUTPUT_SCHEMA_VERSION,
        "resume_boundary": CHECKPOINT_RESUME_BOUNDARY,
        "status": "running",
        "tasks": list(task_names),
        "num_rounds": int(args.num_rounds),
        "num_clients": int(NUM_CLIENTS),
        "seed": int(args.seed),
        "partition": _json_safe(partition_config.__dict__),
        "data_root": str(Path(args.data_root).resolve()),
        "synthetic_datasets": sorted(args.synthetic_datasets),
        "download_datasets": bool(args.download_datasets),
        "task_seed_policy": "seed + task index",
        "run_identity": run_identity,
        "run_identity_hash": _stable_hash(run_identity),
        "dataset_manifest": _json_safe(dataset_manifest),
        "dataset_provenance": _stable_dataset_provenance(dataset_manifest),
        "source_revision": _source_revision(),
        "completed_tasks": [],
        "task_results": [],
    }
    if checkpoint is not None:
        _validate_checkpoint_identity(
            checkpoint,
            task_names=task_names,
            args=args,
            partition_config=partition_config,
            dataset_manifest=dataset_manifest,
        )
        if checkpoint.get("dataset_provenance") != identity["dataset_provenance"]:
            raise ValueError("Checkpoint dataset provenance differs from current datasets; refusing resume")
        identity = checkpoint
    else:
        _atomic_json_write(output_dir / "checkpoint.json", identity)
    _atomic_json_write(output_dir / "dataset_manifest.json", identity["dataset_manifest"])

    task_results = list(identity.get("task_results", []))
    label_records_all = []
    label_payloads = {}
    measurement_rows = []

    for prior in task_results:
        label_records_all.extend(prior.get("label_records", []))
        label_payloads[prior["task"]] = prior.get("label_payload", {})
        measurement_rows.extend(prior.get("round_rows", []))
    completed_tasks = set(identity.get("completed_tasks", []))

    for task_index, (task_name, model_fn, trainset, testloader) in enumerate(experiments):
        if task_name in completed_tasks:
            print(f"  Resuming: skipping completed task {task_name}")
            continue
        # Task-local seeding makes an interrupted task repeat independently of
        # how many earlier tasks were skipped during a resume.
        set_reproducibility_seed(args.seed + task_index)
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
        identity.update(
            {
                "completed_tasks": [result_item["task"] for result_item in task_results],
                "task_results": _json_safe(task_results),
            }
        )
        _atomic_json_write(output_dir / "checkpoint.json", identity)
        # Checkpoint first: derived output failures must not lose completed work.
        save_label_distribution_outputs(label_records_all, label_payloads, output_dir)
        _atomic_dataframe_to_csv(
            pd.DataFrame(measurement_rows),
            output_dir / "per_round_client_measurements.csv",
        )
        save_accuracy_curves(task_results, output_dir)

    label_df = save_label_distribution_outputs(
        label_records_all,
        label_payloads,
        output_dir,
    )
    measurements = pd.DataFrame(measurement_rows)
    _atomic_dataframe_to_csv(measurements, output_dir / "per_round_client_measurements.csv")
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
        "dataset_manifest_file": "dataset_manifest.json",
        "dataset_manifest": dataset_manifest,
        "dataset_provenance": {
            result["task"]: dataset_manifest["datasets"][result["task"]]
            for result in task_results
        },
        "environment": environment_manifest(),
        "outputs": {
            "label_summary_csv": "label_distribution_summary.csv",
            "label_raw_json": "label_distribution_raw.json",
            "per_round_measurements_csv": "per_round_client_measurements.csv",
            "accuracy_curves_csv": "accuracy_curves.csv",
            "correlations_csv": "signal_contribution_correlations.csv",
            "controlled_regression_csv": "controlled_regression.csv",
            "figures_dir": "figures/",
            "progress_jsonl": "progress.jsonl",
        },
    }
    identity["status"] = "completed"
    identity["completed_tasks"] = [result_item["task"] for result_item in task_results]
    identity["task_results"] = _json_safe(task_results)
    _atomic_json_write(output_dir / "checkpoint.json", identity)
    _atomic_json_write(output_dir / "manifest.json", manifest)
    _record_progress(output_dir, "run_completed", measurement_rows=len(measurements))

    print("\n=== Experiment 1 Complete ===")
    print(f"Label rows: {len(label_df)}")
    print(f"Measurement rows: {len(measurements)}")
    print(f"Correlation rows: {len(corr_df)}")
    print(f"Regression rows: {len(reg_df)}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
