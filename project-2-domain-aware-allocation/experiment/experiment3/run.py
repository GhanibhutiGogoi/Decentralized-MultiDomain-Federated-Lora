"""Project 2 Experiment 3 production orchestration.

Experiment 3 runs three paired aggregation arms over the same task, seed, and
fresh partition:

* baseline: ``w_i q_i``
* Form A: ``w_i q_i lambda_A,i``
* Form B: ``w_i q_i lambda_B,i``

The Experiment 2 calibration bundle remains a runtime input boundary. Without a
valid production bundle this runner fails before preparing or cleaning outputs.
Tests may inject tiny in-memory runtime operations into this same orchestration
path; there is no separate dry-run runner or output schema.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

PROJECT2_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_CODE_ROOT = PROJECT2_ROOT / "experiment"
for path in (PROJECT2_ROOT, EXPERIMENT_CODE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from experiment2.lambda_aggregation import normalized_aggregation_weights  # noqa: E402
from experiment3.arms import ARMS, ClientRecord, calculate_lambda_weights  # noqa: E402
from experiment3.calibration_bundle import CalibrationBundle, load_calibration_bundle  # noqa: E402
from experiment3.checkpoint import (  # noqa: E402
    RunIdentity,
    atomic_write_json,
    checkpoint_payload,
    read_checkpoint,
    stable_hash,
    write_checkpoint,
)
from experiment3.metrics import (  # noqa: E402
    fairness_spread_variance,
    per_domain_accuracy,
    within_round_rank_agreement,
    worst_domain_accuracy,
)
from experiment3.seeds import Experiment3Seeds, derive_seeds  # noqa: E402
from experiment3.statistics import (  # noqa: E402
    compare_to_mde,
    confidence_interval,
    paired_arm_differences,
    paired_permutation_test,
    pooled_summary,
    task_level_summary,
)
from framework.datasets import DEFAULT_DATA_ROOT  # noqa: E402
from framework.utils import (  # noqa: E402
    ensure_cleanup_within_allowed_root,
    ensure_disjoint_directory,
    ensure_not_cleanup_parent,
    prepare_output_directory,
    set_reproducibility_seed,
)


OUTPUT_ROOT = PROJECT2_ROOT / "outputs"
EXP1_DIR = OUTPUT_ROOT / "exp1"
EXP2_DIR = OUTPUT_ROOT / "exp2"
OUTPUT_DIR = OUTPUT_ROOT / "exp3"
DEFAULT_TASKS = (
    "CIFAR-CNN",
    "Fashion-MLP",
    "AGNews-LSTM",
    "Tabular-MLP",
    "Audio-1DCNN",
)


class Experiment3RunnerError(RuntimeError):
    """Raised when Experiment 3 orchestration cannot proceed."""


@dataclass(frozen=True)
class Experiment3Config:
    """Complete Experiment 3 execution configuration."""

    calibration_bundle_path: Path
    output_dir: Path = OUTPUT_DIR
    tasks: tuple[str, ...] = DEFAULT_TASKS
    experiment1_run_id: str = ""
    experiment2_run_id: str = ""
    run_seed: int = 42
    rounds: int = 1
    clients: int = 3
    partition: str = "dirichlet"
    partition_alpha: float = 0.5
    overwrite: bool = False
    resume: bool = False
    data_root: Path = DEFAULT_DATA_ROOT
    download_datasets: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    target_accuracy: float | None = None
    mde: float | None = None
    allowed_tasks: tuple[str, ...] = field(default=DEFAULT_TASKS, repr=False)


@dataclass(frozen=True)
class TaskRuntime:
    """Runtime task loaded by the production or injected runtime backend."""

    name: str
    payload: Any


@dataclass(frozen=True)
class PreparedTask:
    """Partitioned task state shared by all three paired arms."""

    partition_hash: str
    client_ids: tuple[str, ...]
    payload: Any


@dataclass(frozen=True)
class ClientUpdate:
    """One trained client update and its calibration features."""

    client_id: str
    state: Any
    sample_count: int
    quality_score: float
    features: Mapping[str, float]
    domain: str
    intended_domain_divergence: float
    metadata: Mapping[str, object] = field(default_factory=dict)


class Experiment3RuntimeOps(Protocol):
    """Operations that bind orchestration to real or injected runtimes."""

    def load_tasks(self, config: Experiment3Config) -> list[TaskRuntime]:
        ...

    def prepare_task(
        self,
        task: TaskRuntime,
        config: Experiment3Config,
        seeds: Experiment3Seeds,
    ) -> PreparedTask:
        ...

    def initial_state(self, task: TaskRuntime, config: Experiment3Config) -> Any:
        ...

    def reset_stochastic_state(
        self,
        task: TaskRuntime,
        prepared: PreparedTask,
        *,
        round_id: int,
        seed: int,
        config: Experiment3Config,
    ) -> None:
        ...

    def train_client_updates(
        self,
        task: TaskRuntime,
        prepared: PreparedTask,
        *,
        arm: str,
        round_id: int,
        global_state: Any,
        config: Experiment3Config,
    ) -> list[ClientUpdate]:
        ...

    def aggregate(
        self,
        task: TaskRuntime,
        prepared: PreparedTask,
        *,
        arm: str,
        round_id: int,
        global_state: Any,
        updates: Sequence[ClientUpdate],
        lambda_weights: Sequence[float] | None,
        config: Experiment3Config,
    ) -> Any:
        ...

    def evaluate(
        self,
        task: TaskRuntime,
        prepared: PreparedTask,
        *,
        arm: str,
        round_id: int,
        global_state: Any,
        config: Experiment3Config,
    ) -> tuple[float, Mapping[str, float]]:
        ...

    def save_state(self, path: Path, state: Any) -> None:
        ...

    def load_state(self, path: Path) -> Any:
        ...


def validate_output_scope(output_dir: Path) -> None:
    """Reject Experiment 3 output paths outside ``outputs/exp3`` ownership."""
    ensure_cleanup_within_allowed_root(
        output_dir,
        OUTPUT_DIR,
        repository_root=PROJECT2_ROOT.parent,
        project_root=PROJECT2_ROOT,
        shared_outputs_root=OUTPUT_ROOT,
        output_label="Experiment 3 output directory",
        allowed_label="Experiment 3 allowed output root",
    )
    ensure_disjoint_directory(
        output_dir,
        EXP1_DIR,
        output_label="Experiment 3 output directory",
        protected_label="Experiment 1 output directory",
    )
    ensure_disjoint_directory(
        output_dir,
        EXP2_DIR,
        output_label="Experiment 3 output directory",
        protected_label="Experiment 2 output directory",
    )
    ensure_not_cleanup_parent(
        output_dir,
        OUTPUT_ROOT,
        output_label="Experiment 3 output directory",
        protected_label="shared Project 2 outputs directory",
    )


def validate_configuration(
    tasks: Sequence[str],
    rounds: int,
    clients: int,
    *,
    allowed_tasks: Sequence[str] = DEFAULT_TASKS,
) -> tuple[str, ...]:
    """Validate task list and run shape before any output cleanup."""
    if not tasks:
        raise Experiment3RunnerError("At least one task is required.")
    unknown = sorted(set(tasks) - set(allowed_tasks))
    if unknown:
        raise Experiment3RunnerError(f"Unknown Experiment 3 task(s): {unknown}.")
    task_tuple = tuple(tasks)
    if len(set(task_tuple)) != len(task_tuple):
        raise Experiment3RunnerError("Duplicate task names are not allowed.")
    if int(rounds) <= 0 or int(clients) <= 0:
        raise Experiment3RunnerError("Rounds and clients must be positive.")
    return task_tuple


def _config_payload(config: Experiment3Config, seeds: Experiment3Seeds) -> dict[str, object]:
    return {
        "tasks": list(config.tasks),
        "experiment1_run_id": config.experiment1_run_id,
        "experiment2_run_id": config.experiment2_run_id,
        "seeds": seeds.to_manifest(),
        "rounds": int(config.rounds),
        "clients": int(config.clients),
        "partition": config.partition,
        "partition_alpha": float(config.partition_alpha),
        "arms": list(ARMS),
        "target_accuracy": config.target_accuracy,
        "mde": config.mde,
        "paired_stochastic_seed_policy": "arm-independent stable hash of run/task/round",
    }


def _sanitize(value: object) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))


def _checkpoint_path(output_dir: Path, task: str, arm: str, seed: int, round_id: int) -> Path:
    return output_dir / "checkpoints" / (
        f"{_sanitize(task)}__{_sanitize(arm)}__seed{seed}__round{round_id}.json"
    )


def _state_path(output_dir: Path, task: str, arm: str, seed: int, round_id: int) -> Path:
    return output_dir / "checkpoints" / (
        f"{_sanitize(task)}__{_sanitize(arm)}__seed{seed}__round{round_id}.state"
    )


def _paired_stochastic_seed(
    *,
    config: Experiment3Config,
    seeds: Experiment3Seeds,
    task_name: str,
    round_id: int,
) -> int:
    payload = {
        "stream": "experiment3-paired-arm-stochastic-state",
        "run_seed": int(config.run_seed),
        "model_seed": int(seeds.model_seed),
        "dataloader_seed": int(seeds.dataloader_seed),
        "task": task_name,
        "round": int(round_id),
        "partition": config.partition,
        "partition_alpha": float(config.partition_alpha),
    }
    return int(stable_hash(payload)[:8], 16)


def _atomic_write_dataframe_csv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            df.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _lambda_weights_for_arm(
    *,
    arm: str,
    records: Sequence[ClientRecord],
    bundle: CalibrationBundle,
) -> list[float] | None:
    if arm == "baseline":
        return None
    return calculate_lambda_weights(records, bundle, arm)


def _records_from_updates(
    task_name: str,
    round_id: int,
    updates: Sequence[ClientUpdate],
) -> list[ClientRecord]:
    return [
        ClientRecord(
            task=task_name,
            round=round_id,
            client_id=update.client_id,
            sample_count=int(update.sample_count),
            quality_score=float(update.quality_score),
            features=update.features,
        )
        for update in updates
    ]


def _divergence_orders(updates: Sequence[ClientUpdate]) -> dict[str, float]:
    values = pd.Series(
        [float(update.intended_domain_divergence) for update in updates],
        index=[update.client_id for update in updates],
        dtype=float,
    )
    return values.rank(method="average").to_dict()


def _round_rows(
    *,
    config: Experiment3Config,
    task: TaskRuntime,
    prepared: PreparedTask,
    arm: str,
    round_id: int,
    updates: Sequence[ClientUpdate],
    lambda_weights: Sequence[float] | None,
    global_accuracy: float,
    domain_accuracy: Mapping[str, float],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    samples = [update.sample_count for update in updates]
    qualities = [update.quality_score for update in updates]
    applied_lambda = (
        [1.0 for _ in updates] if lambda_weights is None else [float(v) for v in lambda_weights]
    )
    realized = normalized_aggregation_weights(
        samples,
        qualities,
        lambda_weights,
        client_ids=[update.client_id for update in updates],
    )
    divergence_rank = _divergence_orders(updates)

    observation_rows = []
    weight_rows = []
    rank_rows = []
    for update, lam, weight in zip(updates, applied_lambda, realized):
        accuracy = float(domain_accuracy[update.client_id])
        observation_rows.append(
            {
                "task": task.name,
                "seed": int(config.run_seed),
                "partition_hash": prepared.partition_hash,
                "arm": arm,
                "round": int(round_id),
                "client_id": update.client_id,
                "domain": update.domain,
                "accuracy": accuracy,
            }
        )
        weight_rows.append(
            {
                "task": task.name,
                "seed": int(config.run_seed),
                "partition_hash": prepared.partition_hash,
                "arm": arm,
                "round": int(round_id),
                "client_id": update.client_id,
                "sample_count": int(update.sample_count),
                "quality_score": float(update.quality_score),
                "lambda_weight": float(lam),
                "realized_aggregation_weight": float(weight),
            }
        )
        rank_rows.append(
            {
                "task": task.name,
                "seed": int(config.run_seed),
                "partition_hash": prepared.partition_hash,
                "arm": arm,
                "round": int(round_id),
                "client_id": update.client_id,
                "lambda_weight": float(lam),
                "intended_domain_divergence_order": float(
                    divergence_rank[update.client_id]
                ),
            }
        )

    global_row = {
        "task": task.name,
        "seed": int(config.run_seed),
        "partition_hash": prepared.partition_hash,
        "arm": arm,
        "round": int(round_id),
        "global_accuracy": float(global_accuracy),
    }
    return observation_rows, weight_rows, rank_rows, global_row


def _final_round_global_accuracy(global_df: pd.DataFrame) -> pd.DataFrame:
    max_round = global_df.groupby(["task", "seed", "partition_hash", "arm"])[
        "round"
    ].transform("max")
    return global_df[global_df["round"] == max_round].reset_index(drop=True)


def _convergence_speed(global_df: pd.DataFrame, target_accuracy: float) -> pd.DataFrame:
    rows = []
    for key, group in global_df.groupby(["task", "seed", "partition_hash", "arm"]):
        reached = group[group["global_accuracy"] >= float(target_accuracy)]
        rows.append(
            {
                "task": key[0],
                "seed": key[1],
                "partition_hash": key[2],
                "arm": key[3],
                "target_accuracy": float(target_accuracy),
                "round_to_target": None
                if reached.empty
                else int(reached.sort_values("round").iloc[0]["round"]),
            }
        )
    return pd.DataFrame(rows)


def _paired_stat_artifacts(final_global: pd.DataFrame, config: Experiment3Config) -> dict[str, pd.DataFrame]:
    paired_frames = []
    permutation_rows = []
    ci_rows = []
    task_summary_frames = []
    mde_rows = []

    for comparison_arm in ("form_a", "form_b"):
        paired = paired_arm_differences(
            final_global,
            baseline_arm="baseline",
            comparison_arm=comparison_arm,
            value_col="global_accuracy",
        )
        paired = paired.copy()
        paired.insert(0, "comparison_arm", comparison_arm)
        paired_frames.append(paired)

        differences = paired["difference"].to_numpy(dtype=float)
        permutation = paired_permutation_test(
            differences,
            seed=int(config.run_seed),
        )
        permutation_rows.append(
            {
                "comparison_arm": comparison_arm,
                "observed_mean_difference": permutation.observed_mean_difference,
                "p_value": permutation.p_value,
                "n_pairs": permutation.n_pairs,
                "method": permutation.method,
                "alternative": permutation.alternative,
                "permutations": permutation.permutations,
            }
        )
        ci_low, ci_high = confidence_interval(differences)
        pooled = pooled_summary(paired)
        ci_rows.append(
            {
                "comparison_arm": comparison_arm,
                "n_pairs": pooled["n_pairs"],
                "mean_difference": pooled["mean_difference"],
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
        task_summary = task_level_summary(paired).copy()
        task_summary.insert(0, "comparison_arm", comparison_arm)
        task_summary_frames.append(task_summary)
        if config.mde is not None:
            mde = compare_to_mde(float(pooled["mean_difference"]), mde=config.mde)
            mde_rows.append(
                {
                    "comparison_arm": comparison_arm,
                    "effect": mde["effect"],
                    "mde": mde["mde"],
                    "meets_mde": mde["meets_mde"],
                }
            )

    artifacts = {
        "paired_arm_differences.csv": pd.concat(paired_frames, ignore_index=True),
        "paired_permutation_tests.csv": pd.DataFrame(permutation_rows),
        "paired_confidence_intervals.csv": pd.DataFrame(ci_rows),
        "paired_task_summaries.csv": pd.concat(task_summary_frames, ignore_index=True),
    }
    if config.mde is not None:
        artifacts["mde_comparison.csv"] = pd.DataFrame(mde_rows)
    return artifacts


def _write_final_artifacts(
    *,
    output_dir: Path,
    observation_rows: list[dict[str, object]],
    global_rows: list[dict[str, object]],
    weight_rows: list[dict[str, object]],
    rank_rows: list[dict[str, object]],
    config: Experiment3Config,
) -> list[str]:
    observations = pd.DataFrame(observation_rows)
    global_df = pd.DataFrame(global_rows)
    weights = pd.DataFrame(weight_rows)
    rank_df = pd.DataFrame(rank_rows)
    final_global = _final_round_global_accuracy(global_df)
    artifacts = {
        "per_round_client_observations.csv": observations,
        "global_accuracy_per_round.csv": global_df,
        "final_round_global_accuracy.csv": final_global,
        "per_domain_accuracy.csv": per_domain_accuracy(observations),
        "worst_domain_accuracy.csv": worst_domain_accuracy(observations),
        "fairness_spread_variance.csv": fairness_spread_variance(observations),
        "realized_aggregation_weights.csv": weights,
        "lambda_rank_agreement.csv": within_round_rank_agreement(rank_df),
    }
    artifacts.update(_paired_stat_artifacts(final_global, config))
    if config.target_accuracy is not None:
        artifacts["convergence_speed.csv"] = _convergence_speed(
            global_df,
            float(config.target_accuracy),
        )

    written = []
    for name, df in artifacts.items():
        _atomic_write_dataframe_csv(df, output_dir / name)
        written.append(name)
    return written


def run_experiment3(
    config: Experiment3Config,
    *,
    runtime_ops: Experiment3RuntimeOps | None = None,
    allow_engineering_fixture: bool = False,
) -> dict[str, object]:
    """Run the production Experiment 3 orchestration.

    ``runtime_ops`` is injectable only for tests. Production callers use the
    default runtime, which loads the actual Project 2 tasks and Project 1
    federated training/aggregation stack.
    """
    task_tuple = validate_configuration(
        config.tasks,
        config.rounds,
        config.clients,
        allowed_tasks=config.allowed_tasks,
    )
    if config.resume and config.overwrite:
        raise Experiment3RunnerError("--resume and --overwrite cannot be combined.")
    if not allow_engineering_fixture and config.mde is None:
        raise Experiment3RunnerError("Scientific Experiment 3 runs must configure --mde.")
    validate_output_scope(config.output_dir)
    seeds = derive_seeds(config.run_seed)
    bundle = load_calibration_bundle(
        config.calibration_bundle_path,
        expected_tasks=task_tuple,
        expected_experiment1_run_id=config.experiment1_run_id,
        expected_experiment2_run_id=config.experiment2_run_id,
        allow_engineering_fixture=allow_engineering_fixture,
    )
    config_payload = _config_payload(config, seeds)
    identity = RunIdentity(
        config_hash=stable_hash(config_payload),
        calibration_bundle_hash=bundle.content_hash,
        source_commit_sha=bundle.source_commit_sha,
    )
    output_dir = prepare_output_directory(
        config.output_dir,
        overwrite=config.overwrite,
        experiment_name="Experiment 3",
        allowed_cleanup_root=OUTPUT_DIR,
        repository_root=PROJECT2_ROOT.parent,
        project_root=PROJECT2_ROOT,
        shared_outputs_root=OUTPUT_ROOT,
    )
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    ops = runtime_ops if runtime_ops is not None else create_default_runtime_ops()
    tasks = ops.load_tasks(config)
    loaded_task_names = tuple(task.name for task in tasks)
    if loaded_task_names != task_tuple:
        raise Experiment3RunnerError(
            "Runtime loaded task order does not match the Experiment 3 configuration."
        )

    observation_rows: list[dict[str, object]] = []
    global_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    partitions = []

    for task in tasks:
        prepared = ops.prepare_task(task, config, seeds)
        if len(prepared.client_ids) != int(config.clients):
            raise Experiment3RunnerError(
                f"Task {task.name!r} produced {len(prepared.client_ids)} clients, "
                f"expected {config.clients}."
            )
        partitions.append(
            {
                "task": task.name,
                "partition_hash": prepared.partition_hash,
                "client_ids": list(prepared.client_ids),
            }
        )

        ops.reset_stochastic_state(
            task,
            prepared,
            round_id=0,
            seed=_paired_stochastic_seed(
                config=config,
                seeds=seeds,
                task_name=task.name,
                round_id=0,
            ),
            config=config,
        )
        initial_state = ops.initial_state(task, config)
        states = {arm: copy.deepcopy(initial_state) for arm in ARMS}
        for arm in ARMS:
            for round_id in range(1, int(config.rounds) + 1):
                cp_path = _checkpoint_path(output_dir, task.name, arm, config.run_seed, round_id)
                if config.resume and cp_path.exists():
                    checkpoint = read_checkpoint(cp_path, expected_identity=identity)
                    data = checkpoint["data"]
                    states[arm] = ops.load_state(output_dir / str(data["state_file"]))
                    observation_rows.extend(data["observation_rows"])
                    weight_rows.extend(data["weight_rows"])
                    rank_rows.extend(data["rank_rows"])
                    global_rows.append(data["global_row"])
                    continue

                ops.reset_stochastic_state(
                    task,
                    prepared,
                    round_id=round_id,
                    seed=_paired_stochastic_seed(
                        config=config,
                        seeds=seeds,
                        task_name=task.name,
                        round_id=round_id,
                    ),
                    config=config,
                )
                updates = ops.train_client_updates(
                    task,
                    prepared,
                    arm=arm,
                    round_id=round_id,
                    global_state=states[arm],
                    config=config,
                )
                if tuple(update.client_id for update in updates) != prepared.client_ids:
                    raise Experiment3RunnerError(
                        f"Task {task.name!r} arm {arm} round {round_id} returned "
                        "client updates out of configured order."
                    )
                records = _records_from_updates(task.name, round_id, updates)
                lambda_weights = _lambda_weights_for_arm(
                    arm=arm,
                    records=records,
                    bundle=bundle,
                )
                next_state = ops.aggregate(
                    task,
                    prepared,
                    arm=arm,
                    round_id=round_id,
                    global_state=states[arm],
                    updates=updates,
                    lambda_weights=lambda_weights,
                    config=config,
                )
                global_accuracy, domain_accuracy = ops.evaluate(
                    task,
                    prepared,
                    arm=arm,
                    round_id=round_id,
                    global_state=next_state,
                    config=config,
                )
                missing_domains = set(prepared.client_ids) - set(domain_accuracy)
                if missing_domains:
                    raise Experiment3RunnerError(
                        f"Missing domain accuracy for clients: {sorted(missing_domains)}"
                    )

                state_file = _state_path(output_dir, task.name, arm, config.run_seed, round_id)
                ops.save_state(state_file, next_state)
                relative_state_file = state_file.relative_to(output_dir)
                obs, weights, ranks, global_row = _round_rows(
                    config=config,
                    task=task,
                    prepared=prepared,
                    arm=arm,
                    round_id=round_id,
                    updates=updates,
                    lambda_weights=lambda_weights,
                    global_accuracy=global_accuracy,
                    domain_accuracy=domain_accuracy,
                )
                payload = checkpoint_payload(
                    identity=identity,
                    task=task.name,
                    arm=arm,
                    seed=config.run_seed,
                    partition_hash=prepared.partition_hash,
                    round_id=round_id,
                    client_id=None,
                    status="round_complete",
                    data={
                        "state_file": str(relative_state_file),
                        "observation_rows": obs,
                        "weight_rows": weights,
                        "rank_rows": ranks,
                        "global_row": global_row,
                    },
                )
                write_checkpoint(cp_path, payload)

                states[arm] = next_state
                observation_rows.extend(obs)
                weight_rows.extend(weights)
                rank_rows.extend(ranks)
                global_rows.append(global_row)

    outputs = _write_final_artifacts(
        output_dir=output_dir,
        observation_rows=observation_rows,
        global_rows=global_rows,
        weight_rows=weight_rows,
        rank_rows=rank_rows,
        config=config,
    )
    manifest = {
        "project": "Project 2",
        "experiment": "Experiment 3",
        "status": "complete",
        "scientific_results": not allow_engineering_fixture,
        "calibration_bundle_file": str(config.calibration_bundle_path),
        "calibration_bundle_hash": bundle.content_hash,
        "calibration_schema_version": bundle.schema_version,
        "source_experiment1_run_id": bundle.source_experiment1_run_id,
        "source_experiment2_run_id": bundle.source_experiment2_run_id,
        "source_commit_sha": bundle.source_commit_sha,
        "run_config": config_payload,
        "identity": asdict(identity),
        "partitions": partitions,
        "outputs": outputs,
        "checkpoint_dir": "checkpoints/",
    }
    write_checkpoint(
        output_dir / "run_manifest.checkpoint.json",
        checkpoint_payload(
            identity=identity,
            task="ALL",
            arm="ALL",
            seed=config.run_seed,
            partition_hash=stable_hash(partitions),
            round_id=int(config.rounds),
            client_id=None,
            status="finalized",
            data={"outputs": outputs},
        ),
    )
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def prepare_validated_run(
    *,
    calibration_bundle_path: Path,
    output_dir: Path,
    tasks: Sequence[str],
    experiment1_run_id: str,
    experiment2_run_id: str,
    run_seed: int,
    rounds: int,
    clients: int,
    mde: float | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Backward-compatible entry point now running the real orchestration."""
    return run_experiment3(
        Experiment3Config(
            calibration_bundle_path=calibration_bundle_path,
            output_dir=output_dir,
            tasks=tuple(tasks),
            experiment1_run_id=experiment1_run_id,
            experiment2_run_id=experiment2_run_id,
            run_seed=run_seed,
            rounds=rounds,
            clients=clients,
            mde=mde,
            overwrite=overwrite,
        )
    )


class Project2RuntimeOps:
    """Runtime adapter for the actual Project 2 models, data, and clients."""

    def __init__(self) -> None:
        from experiment1.project1_bridge import activate_project1_imports

        activate_project1_imports()
        import torch
        import torch.nn as nn
        from config import (  # type: ignore
            BATCH_TO_MAX_RANK,
            CLIENT_BATCH_SIZES,
            CLIENT_EPOCHS,
            FIXED_RANK,
            LORA_A_SUFFIXES,
            LORA_B_SUFFIXES,
            LORA_SUFFIXES,
        )
        from Federated.client import compute_quality_score, train_client
        from Federated.fedavg_aggregation import fedavg_quality_weighted
        from Federated.utilities import evaluate
        from rank_allocation.LoRa_rank_projection import load_global_state
        from rank_allocation.rank_selector import estimate_optimal_rank
        from experiment1.contribution import flatten_update_vector
        from experiment1.run import load_experiments
        from experiment1.signals import label_distribution_records, update_dissimilarity_records
        from framework.partitioning import PartitionConfig, make_client_loaders
        from experiment2.lambda_aggregation import fedavg_quality_lambda_weighted

        self.torch = torch
        self.nn = nn
        self.batch_to_max_rank = BATCH_TO_MAX_RANK
        self.client_batch_sizes = tuple(CLIENT_BATCH_SIZES)
        self.client_epochs = CLIENT_EPOCHS
        self.fixed_rank = FIXED_RANK
        self.lora_a_suffixes = LORA_A_SUFFIXES
        self.lora_b_suffixes = LORA_B_SUFFIXES
        self.lora_suffixes = LORA_SUFFIXES
        self.compute_quality_score = compute_quality_score
        self.train_client = train_client
        self.fedavg_quality_weighted = fedavg_quality_weighted
        self.evaluate_fn = evaluate
        self.load_global_state = load_global_state
        self.estimate_optimal_rank = estimate_optimal_rank
        self.flatten_update_vector = flatten_update_vector
        self.load_experiments = load_experiments
        self.label_distribution_records = label_distribution_records
        self.update_dissimilarity_records = update_dissimilarity_records
        self.PartitionConfig = PartitionConfig
        self.make_client_loaders = make_client_loaders
        self.fedavg_quality_lambda_weighted = fedavg_quality_lambda_weighted
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _batch_sizes(self, config: Experiment3Config) -> tuple[int, ...]:
        if int(config.clients) > len(self.client_batch_sizes):
            raise Experiment3RunnerError(
                "Configured clients exceed Project 1 CLIENT_BATCH_SIZES."
            )
        return self.client_batch_sizes[: int(config.clients)]

    def load_tasks(self, config: Experiment3Config) -> list[TaskRuntime]:
        experiments, _ = self.load_experiments(
            task_names=list(config.tasks),
            data_root=config.data_root,
            download_datasets=config.download_datasets,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            synthetic_datasets=set(),
        )
        return [
            TaskRuntime(
                name=task_name,
                payload={
                    "model_fn": model_fn,
                    "trainset": trainset,
                    "testloader": testloader,
                },
            )
            for task_name, model_fn, trainset, testloader in experiments
        ]

    def prepare_task(
        self,
        task: TaskRuntime,
        config: Experiment3Config,
        seeds: Experiment3Seeds,
    ) -> PreparedTask:
        partition_config = self.PartitionConfig(
            strategy=config.partition,
            alpha=float(config.partition_alpha),
            seed=int(seeds.partition_seed),
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        loaders, metadata = self.make_client_loaders(
            task.payload["trainset"],
            self._batch_sizes(config),
            partition_config,
        )
        label_records, _ = self.label_distribution_records(
            task_name=task.name,
            labels=metadata["labels"],
            indices_by_client=metadata["indices_by_client"],
            num_classes=metadata["num_classes"],
            is_synthetic=bool(getattr(task.payload["trainset"], "is_synthetic", False)),
        )
        import hashlib

        partition_payload = {
            "task": task.name,
            "strategy": config.partition,
            "alpha": float(config.partition_alpha),
            "seed": int(seeds.partition_seed),
            "indices_by_client": metadata["indices_by_client"],
        }
        partition_hash = hashlib.sha256(
            json.dumps(partition_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return PreparedTask(
            partition_hash=partition_hash,
            client_ids=tuple(str(idx) for idx in range(len(loaders))),
            payload={
                "loaders": loaders,
                "label_by_client": {
                    str(record["client_id"]): record for record in label_records
                },
            },
        )

    def initial_state(self, task: TaskRuntime, config: Experiment3Config) -> Any:
        model = task.payload["model_fn"](self.fixed_rank).to(self.device)
        return model.state_dict()

    def reset_stochastic_state(
        self,
        task: TaskRuntime,
        prepared: PreparedTask,
        *,
        round_id: int,
        seed: int,
        config: Experiment3Config,
    ) -> None:
        del task, round_id, config
        set_reproducibility_seed(int(seed))
        for client_idx, loader in enumerate(prepared.payload.get("loaders", [])):
            generator = getattr(loader, "generator", None)
            if isinstance(generator, self.torch.Generator):
                generator.manual_seed(int(seed) + client_idx)

    def train_client_updates(
        self,
        task: TaskRuntime,
        prepared: PreparedTask,
        *,
        arm: str,
        round_id: int,
        global_state: Any,
        config: Experiment3Config,
    ) -> list[ClientUpdate]:
        loaders = prepared.payload["loaders"]
        label_by_client = prepared.payload["label_by_client"]
        model_fn = task.payload["model_fn"]
        batch_sizes = self._batch_sizes(config)
        loss_fn = self.nn.CrossEntropyLoss()
        update_vectors = []
        partial: list[dict[str, object]] = []

        for client_idx, loader in enumerate(loaders):
            batch_size = batch_sizes[client_idx]
            probe_rank = self.batch_to_max_rank[batch_size]
            probe = model_fn(probe_rank).to(self.device)
            self.load_global_state(probe, global_state)
            chosen_rank = self.estimate_optimal_rank(probe, loader, loss_fn, batch_size)

            local = model_fn(chosen_rank).to(self.device)
            self.load_global_state(local, global_state)
            state, sample_count = self.train_client(
                local,
                loader,
                self.client_epochs,
                self.device,
            )
            quality = self.compute_quality_score(local, loader, loss_fn, self.device)
            update_vectors.append(
                self.flatten_update_vector(
                    state,
                    global_state,
                    self.lora_a_suffixes,
                    self.lora_b_suffixes,
                    self.lora_suffixes,
                )
            )
            partial.append(
                {
                    "client_id": str(client_idx),
                    "state": state,
                    "sample_count": int(sample_count),
                    "quality_score": float(quality),
                    "chosen_rank": int(chosen_rank),
                }
            )

        update_records = self.update_dissimilarity_records(
            task.name,
            round_id,
            update_vectors,
        )
        update_by_client = {
            str(record["client_id"]): record for record in update_records
        }
        out = []
        for item in partial:
            client_id = str(item["client_id"])
            label_row = label_by_client[client_id]
            update_row = update_by_client[client_id]
            update_l2 = float(update_row["update_l2_distance_to_mean"])
            imbalance = float(label_row["class_imbalance_ratio"])
            features = {
                "log_update_l2": float(np.log1p(max(update_l2, 0.0))),
                "js_to_global": float(label_row["js_to_global"]),
                "update_cosine_distance_to_mean": float(
                    update_row["update_cosine_distance_to_mean"]
                ),
                "normalized_entropy": float(label_row["normalized_entropy"]),
                "log_class_imbalance_ratio": float(np.log1p(max(imbalance, 0.0))),
            }
            out.append(
                ClientUpdate(
                    client_id=client_id,
                    state=item["state"],
                    sample_count=int(item["sample_count"]),
                    quality_score=float(item["quality_score"]),
                    features=features,
                    domain=client_id,
                    intended_domain_divergence=features["js_to_global"],
                    metadata={"chosen_rank": item["chosen_rank"]},
                )
            )
        return out

    def aggregate(
        self,
        task: TaskRuntime,
        prepared: PreparedTask,
        *,
        arm: str,
        round_id: int,
        global_state: Any,
        updates: Sequence[ClientUpdate],
        lambda_weights: Sequence[float] | None,
        config: Experiment3Config,
    ) -> Any:
        ref_sd = task.payload["model_fn"](self.fixed_rank).to(self.device).state_dict()
        return self.fedavg_quality_lambda_weighted(
            [update.state for update in updates],
            [update.sample_count for update in updates],
            [update.quality_score for update in updates],
            self.fixed_rank,
            ref_sd,
            self.device,
            lambda_weights=lambda_weights,
            base_aggregate_fn=self.fedavg_quality_weighted,
        )

    def evaluate(
        self,
        task: TaskRuntime,
        prepared: PreparedTask,
        *,
        arm: str,
        round_id: int,
        global_state: Any,
        config: Experiment3Config,
    ) -> tuple[float, Mapping[str, float]]:
        model = task.payload["model_fn"](self.fixed_rank).to(self.device)
        self.load_global_state(model, global_state)
        global_accuracy = self.evaluate_fn(model, task.payload["testloader"], self.device)
        domain_accuracy = {}
        for client_id, loader in zip(prepared.client_ids, prepared.payload["loaders"]):
            domain_accuracy[client_id] = self.evaluate_fn(model, loader, self.device)
        return float(global_accuracy), domain_accuracy

    def save_state(self, path: Path, state: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        os.close(fd)
        try:
            self.torch.save(state, tmp_name)
            with open(tmp_name, "rb") as handle:
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise

    def load_state(self, path: Path) -> Any:
        return self.torch.load(path, map_location=self.device)


def create_default_runtime_ops() -> Experiment3RuntimeOps:
    """Create the runtime adapter for real Project 2 Experiment 3 execution."""
    return Project2RuntimeOps()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--experiment1-run-id", required=True)
    parser.add_argument("--experiment2-run-id", required=True)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--partition", default="dirichlet")
    parser.add_argument("--partition-alpha", type=float, default=0.5)
    parser.add_argument("--target-accuracy", type=float, default=None)
    parser.add_argument("--mde", type=float, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--download-datasets", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clean only outputs/exp3 or its descendants after validation succeeds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment3(
        Experiment3Config(
            calibration_bundle_path=args.calibration_bundle,
            output_dir=args.output_dir,
            tasks=tuple(args.tasks),
            experiment1_run_id=args.experiment1_run_id,
            experiment2_run_id=args.experiment2_run_id,
            run_seed=args.run_seed,
            rounds=args.rounds,
            clients=args.clients,
            partition=args.partition,
            partition_alpha=args.partition_alpha,
            overwrite=args.overwrite,
            resume=args.resume,
            data_root=args.data_root,
            download_datasets=args.download_datasets,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            target_accuracy=args.target_accuracy,
            mde=args.mde,
        )
    )


if __name__ == "__main__":
    main()
