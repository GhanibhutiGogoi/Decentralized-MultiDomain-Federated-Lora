#!/usr/bin/env python3
"""Audit and summarize the RoBERTa/GLUE quantity-skew comparison on gpu003.

Usage (run on the authorized SSH machine, never as a local experiment)::

    ~/ahlora-venv/bin/python scripts/summarize_quantity_benchmark.py \
        --inputs /path/to/quantity-results /path/to/equal-anchor \
        --output /path/to/quantity-report

Each run directory contains config.json, environment.json, split.json,
graph.json, rounds.jsonl, and (when complete) summary.json. Recursively searches
input directories; an individual run directory is also accepted. Validation
labels come from split.json['validation_labels']. The standalone evaluator's
independent_checkpoint_audit.json is checked against best/final rounds, metrics,
raw-prediction-match flags, checkpoint hashes, source snapshot, and split hash.
Legacy fresh_process_verification.json can verify the final checkpoint only.

Writes per_run.csv, per_round.csv, summary.json, SUMMARY.md, and PDF/PNG figures.
Metrics are independently recomputed from saved predictions and public labeled
validation labels; this is an arithmetic audit, not independent model inference.
Independent checkpoint inference is reported separately and never invented.
Checkpoint files may be omitted from an archived report when exact hashes are
bound to a passing independent audit; any checkpoint files present are hashed.
Incomplete and invalid runs remain in the inventory but do not enter means or
paired effects. Duplicate complete arm/seed runs in a comparison group are
rejected: choose explicit input directories instead of averaging reruns.

Comparisons require identical non-operational config, source snapshot, model
and dataset fingerprints, actual training steps/exposures, and task endpoint.
Equal-size paper-setting anchors and quantity extensions are separate groups.
Within-seed comparisons additionally require identical shards, graph/capacity
assignments, and actual local sample streams. Rank/weight/merge policy are the
deliberate method differences and do not prevent those comparisons.

max_train>0 or rounds<20 means smoke. Other runs are full configured-budget
experiments, not automatically exact paper reproductions. SST-2 best validation
accuracy is primary; final accuracy is secondary. MRPC uses best validation F1
if requested by that run. All scores are percentages. Seed-paired effects are
percentage-point differences; n>=2 gets sample SD and an unadjusted two-sided
Student-t 95% interval. Small-n intervals are exploratory, not evidence of
equivalence, a hidden-test result, or a win over printed paper scores.

Dependencies: numpy, scipy (intervals), matplotlib (figures). The script reads
only artifacts; it neither trains nor modifies checkpoints or paper files.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import socket
import sys

import numpy as np


LABELS = {
    "declora16": "Dec-LoRA reimplementation, rank 16",
    "declora4": "Dec-LoRA reimplementation, rank 4",
    "product16_sample": "Effective products, rank 16, sample weights",
    "fixed_uniform": "Fixed heterogeneous ranks, uniform weights",
    "fixed_sample": "Fixed heterogeneous ranks, sample weights",
    "adaptive_uniform": "Adaptive ranks, uniform weights",
    "adaptive_sample": "Adaptive ranks, sample weights",
}
COLORS = dict(zip(LABELS, ("#202020", "#777777", "#0072B2", "#009E73", "#56B4E9", "#CC79A7", "#D55E00")))
EFFECTS = {
    "primary_adaptive_sample_minus_declora16": {"adaptive_sample": 1, "declora16": -1},
    "adaptive_sample_minus_feasible_declora4": {"adaptive_sample": 1, "declora4": -1},
    "adaptive_sample_minus_product16_sample": {"adaptive_sample": 1, "product16_sample": -1},
    "adaptation_at_uniform_weights": {"adaptive_uniform": 1, "fixed_uniform": -1},
    "adaptation_at_sample_weights": {"adaptive_sample": 1, "fixed_sample": -1},
    "sample_weighting_at_fixed_ranks": {"fixed_sample": 1, "fixed_uniform": -1},
    "sample_weighting_at_adaptive_ranks": {"adaptive_sample": 1, "adaptive_uniform": -1},
    "adaptation_main_effect": {"adaptive_sample": .5, "fixed_sample": -.5, "adaptive_uniform": .5, "fixed_uniform": -.5},
    "sample_weighting_main_effect": {"adaptive_sample": .5, "adaptive_uniform": -.5, "fixed_sample": .5, "fixed_uniform": -.5},
    "adaptation_by_weighting_interaction": {"adaptive_sample": 1, "adaptive_uniform": -1, "fixed_sample": -1, "fixed_uniform": 1},
}
OPERATIONAL_CONFIG = {"assets", "output", "arm", "seed", "threads", "verify_only"}


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def close(observed, expected, description, tolerance=1e-9):
    if not math.isfinite(float(observed)) or abs(float(observed) - float(expected)) > tolerance:
        raise ValueError(f"{description}: observed {observed}, expected {expected}")


def independent_metrics(predictions, labels):
    pred, truth = np.asarray(predictions), np.asarray(labels)
    if (pred.shape != truth.shape or pred.ndim != 1 or not pred.size
            or pred.dtype.kind not in "iu" or truth.dtype.kind not in "iu"
            or not np.isin(pred, [0, 1]).all() or not np.isin(truth, [0, 1]).all()):
        raise ValueError("saved predictions and labels must be same-length nonempty binary integer arrays")
    correct = int(np.count_nonzero(pred == truth))
    tp = int(np.count_nonzero((pred == 1) & (truth == 1)))
    fp = int(np.count_nonzero((pred == 1) & (truth == 0)))
    fn = int(np.count_nonzero((pred == 0) & (truth == 1)))
    denominator = 2 * tp + fp + fn
    return {"accuracy": 100 * correct / len(truth), "f1": 200 * tp / denominator if denominator else 0.,
            "correct": correct, "n": len(truth), "tp": tp, "fp": fp, "fn": fn,
            "predictions_sha256": hashlib.sha256(pred.astype("<i8").tobytes()).hexdigest()}


def check_metric_record(record, metrics, context):
    for field in ("accuracy", "f1", "correct", "n"):
        close(record[field], metrics[field], f"{context}/{field}")
    if record.get("predictions_sha256") != metrics["predictions_sha256"]:
        raise ValueError(f"{context}: prediction hash does not match prediction array")


def check_transport(ledger, graph, context):
    transfers = ledger.get("transfers", [])
    if ledger["messages"] != len(transfers):
        raise ValueError(f"{context}: message count disagrees with edge ledger")
    for row in transfers:
        if row["receiver"] not in graph["neighbors"][str(row["sender"])]:
            raise ValueError(f"{context}: transfer uses an undeclared edge")
        if row["payload_bytes"] != row["tensor_bytes"] + row["metadata_bytes"]:
            raise ValueError(f"{context}: tensor/metadata bytes do not sum to serialized bytes")
    for field in ("bytes", "tensor_bytes", "metadata_bytes"):
        source = "payload_bytes" if field == "bytes" else field
        if ledger[field] != sum(row[source] for row in transfers):
            raise ValueError(f"{context}: {field} differs from explicit transfer sum")


def stats(per_seed):
    values = np.asarray(list(per_seed.values()), dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("statistics need at least one finite observation")
    mean = float(values.mean())
    result = {"n": len(values), "mean": mean, "sample_sd": None, "ci95": None,
              "per_seed": {str(seed): float(value) for seed, value in per_seed.items()},
              "interpretation": "single-seed exploratory observation; no estimated between-seed uncertainty"}
    if len(values) >= 2:
        from scipy.stats import t
        sd = float(values.std(ddof=1))
        critical = float(t.ppf(.975, len(values) - 1))
        halfwidth = critical * sd / math.sqrt(len(values))
        result.update(sample_sd=sd, ci95=[mean - halfwidth, mean + halfwidth],
                      degrees_of_freedom=len(values) - 1, critical_t=critical,
                      interpretation="exploratory unadjusted Student-t interval across seeds; no parity or confirmatory superiority claim")
    return result


def run_directories(inputs):
    folders = set()
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"input does not exist: {path}")
        if (path / "config.json").exists():
            folders.add(path)
        for config in path.rglob("config.json"):
            if "source" not in config.relative_to(path).parts:
                folders.add(config.parent)
    return sorted(folders)


def load_run(folder):
    config = read_json(folder / "config.json")
    run = {"path": str(folder), "arm": config.get("arm"), "seed": config.get("seed"),
           "task": config.get("task"), "partition": config.get("partition"),
           "status": "incomplete", "config": config, "round_records": [], "warnings": []}
    run["classification"] = "smoke" if config.get("max_train", 0) > 0 or config.get("rounds", 0) < 20 else "full_configured_budget"
    run["track"] = "equal_size_paper_setting_anchor" if config.get("partition") == "equal" else "quantity_skew_extension"
    required = ["environment.json", "split.json", "graph.json", "rounds.jsonl"]
    missing = [name for name in required if not (folder / name).exists()]
    if missing:
        run["warnings"].append("Missing run artifacts: " + ", ".join(missing))
        return run
    environment, split, graph = [read_json(folder / name) for name in required[:3]]
    if config.get("task") not in {"sst2", "mrpc"}:
        raise ValueError("only binary SST-2 accuracy and MRPC F1 endpoints are supported")
    if config.get("partition") not in {"equal", "quantity"}:
        raise ValueError("partition must explicitly identify equal or quantity")
    labels = split.get("validation_labels")
    if labels is None:
        raise ValueError("split.json lacks validation_labels; independent prediction-metric audit is impossible")
    source_sha = environment.get("source_sha256")
    if not source_sha or not environment.get("model_revision") or not environment.get("model_files"):
        raise ValueError("source and base-model fingerprints are required for scientific grouping")
    if (folder / "source_manifest.json").exists():
        source_manifest = read_json(folder / "source_manifest.json")
        actual_sha = hashlib.sha256(json.dumps(source_manifest, sort_keys=True).encode()).hexdigest()
        if actual_sha != source_sha:
            raise ValueError("source manifest disagrees with recorded source snapshot hash")
    else:
        raise ValueError("source_manifest.json is required")
    data_files = split.get("data_files", {})
    if set(data_files) != {"train", "validation"} or any(not row.get("file_sha256") for row in data_files.values()):
        raise ValueError("training and validation dataset file fingerprints are required")
    if len(labels) != data_files["validation"]["rows_used"]:
        raise ValueError("validation label count disagrees with dataset manifest")
    counts = np.asarray(split["client_counts"], dtype=int)
    if np.any(counts <= 0) or counts.sum() != split["n_examples"] or len(counts) != split["n_clients"]:
        raise ValueError("invalid client quantity totals")
    if not split.get("proportional_class_allocation", False):
        raise ValueError("quantity benchmark split does not pass proportional class allocation check")
    memberships = {str(cid): sorted(split["clients"][str(cid)]["indices"]) for cid in range(len(counts))}
    ownership = [index for indices in memberships.values() for index in indices]
    if sorted(ownership) != list(range(split["n_examples"])):
        raise ValueError("client shards do not form an exact disjoint training union")
    if digest(memberships) != split["partition_sha256"]:
        raise ValueError("split hash disagrees with stored shard memberships")
    if any(len(memberships[str(cid)]) != count for cid, count in enumerate(counts)):
        raise ValueError("stored membership lengths disagree with client counts")
    weights, matrix = np.asarray(graph["stationary_weights"]), np.asarray(graph["matrix"])
    if (weights.shape != counts.shape or matrix.shape != (len(counts), len(counts))
            or not np.isfinite(matrix).all() or np.any(matrix < 0)):
        raise ValueError("invalid graph weights or matrix")
    if not np.allclose(matrix.sum(1), 1, atol=1e-12, rtol=0) or not np.allclose(weights @ matrix, weights, atol=1e-12, rtol=0):
        raise ValueError("graph does not preserve its stated stationary weights")
    if not np.allclose(weights[:, None] * matrix, weights[None, :] * matrix.T, atol=1e-12, rtol=0):
        raise ValueError("weighted mixer does not satisfy detailed balance")
    for receiver in range(len(counts)):
        for sender in range(len(counts)):
            if sender != receiver and matrix[receiver, sender] > 0 and sender not in graph["neighbors"][str(receiver)]:
                raise ValueError("mixing matrix has support outside the declared graph")
    expected_weights = counts / counts.sum() if config["arm"].endswith("sample") else np.full(len(counts), 1 / len(counts))
    if not np.allclose(weights, expected_weights, atol=1e-12, rtol=0):
        raise ValueError("arm name, dataset quantities, and actual objective weights disagree")
    primary_metric = "accuracy" if config["task"] == "sst2" else "f1"
    raw_rounds = []
    lines = (folder / "rounds.jsonl").read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            raw_rounds.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not (folder / "summary.json").exists():
                run["warnings"].append("Ignored an unfinished final JSONL line from an active run.")
                break
            raise
    training_bytes = evaluation_bytes = total_steps = total_examples = 0
    sample_streams = []
    for expected_round, row in enumerate(raw_rounds, 1):
        if row["round"] != expected_round:
            raise ValueError("round records must be consecutive starting at one")
        metric = independent_metrics(row["validation"]["predictions"], labels)
        check_metric_record(row["validation"], metric, f"round {expected_round}")
        peers = sorted(row["local"], key=int)
        if peers != [str(i) for i in range(len(counts))]:
            raise ValueError("local training records do not cover all peers")
        local_steps = sum(row["local"][cid]["steps"] for cid in peers)
        examples = sum(row["local"][cid]["examples"] for cid in peers)
        if config.get("local_steps", 0):
            for cid in peers:
                if (row["local"][cid]["steps"] != config["local_steps"]
                        or row["local"][cid]["examples"] != config["local_steps"] * config["batch_size"]):
                    raise ValueError("fixed-step run violates its equal per-peer work budget")
        else:
            for cid in peers:
                expected_examples = int(counts[int(cid)]) * config["local_epochs"]
                expected_steps = math.ceil(int(counts[int(cid)]) / config["batch_size"]) * config["local_epochs"]
                if row["local"][cid]["examples"] != expected_examples or row["local"][cid]["steps"] != expected_steps:
                    raise ValueError("local-epoch run violates its recorded client quantities")
        ranks = row["ranks"]
        if len(ranks) != len(counts) or any(isinstance(r, bool) or not isinstance(r, int) or r < 1 for r in ranks):
            raise ValueError("invalid per-peer rank trace")
        cap_violations = sum(rank > cap for rank, cap in zip(ranks, graph["capacity"]))
        if cap_violations and config["arm"] not in {"declora16", "product16_sample"}:
            raise ValueError("resource-constrained arm exceeds a declared rank cap")
        check_transport(row["gossip"], graph, f"round {expected_round} training")
        check_transport(row["evaluation_assembly"], graph, f"round {expected_round} assembly")
        if row["evaluation_assembly"]["root_id"] != graph["root"]:
            raise ValueError("assembly root differs from its predeclared peer")
        training_bytes += row["gossip"]["bytes"]
        evaluation_bytes += row["evaluation_assembly"]["bytes"]
        close(row["cumulative_training_bytes"], training_bytes, "cumulative training bytes", 0)
        close(row["cumulative_evaluation_assembly_bytes"], evaluation_bytes, "cumulative evaluation bytes", 0)
        total_steps += local_steps
        total_examples += examples
        sample_streams.append({cid: {key: row["local"][cid][key] for key in ("steps", "examples", "sample_order_sha256")} for cid in peers})
        run["round_records"].append({
            "round": expected_round, "accuracy": metric["accuracy"], "f1": metric["f1"],
            "correct": metric["correct"], "n_validation": metric["n"],
            "prediction_sha256": metric["predictions_sha256"],
            "cumulative_training_bytes": training_bytes, "cumulative_evaluation_assembly_bytes": evaluation_bytes,
            "round_training_bytes": row["gossip"]["bytes"], "round_evaluation_assembly_bytes": row["evaluation_assembly"]["bytes"],
            "training_steps": local_steps, "training_examples": examples,
            "cumulative_training_steps": total_steps, "cumulative_training_examples": total_examples,
            "probe_examples": sum(value["before"]["probe_examples"] * 2 for value in row.get("probes", {}).values()),
            "rank_example_products": sum(ranks[int(cid)] * row["local"][cid]["examples"] for cid in peers),
            "ranks": ranks, "mean_rank": float(np.mean(ranks)), "min_rank": min(ranks), "max_rank": max(ranks),
            "rank_cap_violations": cap_violations,
            "sample_weighted_training_loss": sum(row["local"][cid]["mean_loss"] * row["local"][cid]["examples"] for cid in peers) / examples,
            "peak_process_cuda_allocated_bytes": row.get("peak_process_cuda_allocated_bytes"),
            "peak_process_cuda_reserved_bytes": row.get("peak_process_cuda_reserved_bytes"),
            "weighted_adapter_disagreement": row.get("weighted_disagreement", {}).get("adapter_variance"),
            "round_seconds": row.get("round_seconds"), "elapsed_seconds": row.get("elapsed_seconds"),
        })
    run.update(primary_metric=primary_metric, source_sha256=source_sha,
               n_train=int(counts.sum()), n_validation=len(labels), n_clients=len(counts),
               total_steps=total_steps, total_examples=total_examples, training_bytes=training_bytes,
               evaluation_assembly_bytes=evaluation_bytes, split_sha256=split["partition_sha256"],
               metric_verification="independent arithmetic from every saved prediction and labeled validation target",
               checkpoint_verification={"status": "missing", "verified_endpoints": [],
                                        "scope": "No independent audit or legacy fresh-process checkpoint record found."})
    summary_path = folder / "summary.json"
    if not summary_path.exists():
        run["warnings"].append("Training is incomplete; excluded from aggregate estimates and paired effects.")
        return run
    summary = read_json(summary_path)
    if summary.get("status") != "complete":
        run["warnings"].append("Run summary does not declare completion.")
        return run
    if len(raw_rounds) != config["rounds"] or summary["rounds"] != config["rounds"]:
        raise ValueError("complete summary does not contain its entire declared round budget")
    if any(summary[key] != config[key] for key in ("arm", "seed", "partition")):
        raise ValueError("summary identity differs from config")
    if summary["source_sha256"] != source_sha or summary["primary_metric"] != primary_metric:
        raise ValueError("summary source/endpoint differs from the run protocol")
    for key, expected in (("training_bytes", training_bytes), ("evaluation_assembly_bytes", evaluation_bytes),
                          ("total_steps", total_steps), ("total_examples", total_examples)):
        close(summary[key], expected, f"summary/{key}", 0)
    final_row = run["round_records"][-1]
    best_row = max(run["round_records"], key=lambda row: row[primary_metric])
    best_metrics = independent_metrics(raw_rounds[best_row["round"] - 1]["validation"]["predictions"], labels)
    final_metrics = independent_metrics(raw_rounds[-1]["validation"]["predictions"], labels)
    check_metric_record(summary["best"]["validation"], best_metrics, "summary best")
    check_metric_record(summary["final"], final_metrics, "summary final")
    if summary["best"]["round"] != best_row["round"]:
        raise ValueError("best checkpoint is not the earliest maximum of the declared validation metric")
    production_bytes = graph["setup_bytes"] + training_bytes + raw_rounds[-1]["evaluation_assembly"]["bytes"]
    close(summary["production_bytes_setup_training_final_assembly"], production_bytes, "production byte total", 0)
    close(summary["all_experiment_bytes_including_evaluations"], graph["setup_bytes"] + training_bytes + evaluation_bytes, "all-experiment byte total", 0)
    audit_path = folder / "independent_checkpoint_audit.json"
    verification_path = folder / "fresh_process_verification.json"
    if audit_path.exists():
        audit = read_json(audit_path)
        if audit.get("status") != "passed" or set(audit.get("checkpoints", {})) != {"best", "final"}:
            raise ValueError("independent checkpoint audit did not pass both best and final checkpoints")
        artifact_checks = audit["artifact_checks"]
        if (artifact_checks["source_sha256"] != source_sha
                or artifact_checks["partition_sha256"] != split["partition_sha256"]
                or artifact_checks["validation_rows"] != len(labels)
                or artifact_checks["partition_examples"] != split["n_examples"]
                or artifact_checks["training_source_indices_sha256"] != data_files["train"]["source_indices_sha256"]
                or not artifact_checks["shards_disjoint_exhaustive"]
                or not artifact_checks["quantity_only_class_quotas_verified"]
                or audit["round_metric_records_verified"] != len(raw_rounds)):
            raise ValueError("independent audit source, split, data ownership, or evaluated round identity disagrees")
        helper_key = "project-3-hierarchical-gossip/src/models/roberta_lora.py"
        if helper_key in source_manifest and audit["model_helper_sha256"] != source_manifest[helper_key]:
            raise ValueError("independent audit used a different model/install helper from the archived run")
        for endpoint, expected_round, expected_metrics in (("best", best_row["round"], best_metrics),
                                                            ("final", final_row["round"], final_metrics)):
            checkpoint_audit = audit["checkpoints"][endpoint]
            if (checkpoint_audit["checkpoint_sha256"] != summary[f"{endpoint}_checkpoint_sha256"]
                    or checkpoint_audit["round"] != expected_round
                    or checkpoint_audit.get("exact_raw_prediction_match") is not True):
                raise ValueError(f"independent {endpoint} checkpoint hash, round or exact prediction match disagrees")
            check_metric_record(checkpoint_audit["metrics"], expected_metrics, f"independent {endpoint} checkpoint audit")
        run["checkpoint_verification"] = {
            "status": "passed", "kind": "standalone_independent_evaluator", "verified_endpoints": ["best", "final"],
            "path": str(audit_path), "audit_sha256": file_digest(audit_path),
            "evaluator_sha256": audit["evaluator_sha256"], "model_helper_sha256": audit["model_helper_sha256"],
            "checkpoint_sha256": {endpoint: audit["checkpoints"][endpoint]["checkpoint_sha256"] for endpoint in ("best", "final")},
            "scope": audit["independence_scope"], "precision": audit.get("precision"),
            "evidence_binding": "Best/final hashes, rounds, metrics and exact raw prediction matches agree with this source-pinned run; local checkpoint files are not required when the audit remains available.",
        }
    elif verification_path.exists():
        verification = read_json(verification_path)
        if not verification.get("matches") or verification["checkpoint_sha256"] != summary["final_checkpoint_sha256"]:
            raise ValueError("fresh-process final checkpoint verification failed or refers to another checkpoint")
        verified_metrics = independent_metrics(verification["metrics"]["predictions"], labels)
        check_metric_record(verification["metrics"], verified_metrics, "fresh-process verification")
        if verified_metrics["predictions_sha256"] != final_metrics["predictions_sha256"]:
            raise ValueError("fresh-process predictions differ from final predictions")
        run["checkpoint_verification"] = {"status": "passed", "kind": "legacy_shared_evaluator_final_only",
            "verified_endpoints": ["final"], "path": str(verification_path),
            "scope": "Saved final checkpoint predictions reproduced in a fresh process; metric arithmetic independently checked here. This does not assert a separately implemented model evaluator."}
        run["warnings"].append("Only the final checkpoint has a legacy fresh-process verification; no independent best-checkpoint inference audit is available.")
    else:
        run["warnings"].append("Missing independent checkpoint audit; no scientific success claim is supported.")
    run["checkpoint_files_present"] = {endpoint: (folder / f"{endpoint}.pt").exists() for endpoint in ("best", "final")}
    for endpoint in ("best", "final"):
        checkpoint = folder / f"{endpoint}.pt"
        if checkpoint.exists() and file_digest(checkpoint) != summary[f"{endpoint}_checkpoint_sha256"]:
            raise ValueError(f"{endpoint} checkpoint bytes differ from summary hash")
    compatibility = {
        "classification": run["classification"], "track": run["track"],
        "config": {key: value for key, value in config.items() if key not in OPERATIONAL_CONFIG},
        "source_sha256": source_sha, "model_revision": environment["model_revision"],
        "model_files": environment["model_files"], "dataset_revision": environment.get("dataset_revision"),
        "data_files": {name: {key: value[key] for key in ("file_sha256", "rows_in_file", "rows_used")} for name, value in data_files.items()},
        "precision": environment.get("precision"), "optimizer_policy": environment.get("optimizer_policy"),
        "primary_metric": primary_metric, "total_training_steps": total_steps, "total_training_examples": total_examples,
    }
    pair_identity = {"partition_sha256": split["partition_sha256"], "label_sha256": split["label_sha256"],
                     "data_indices": {name: value["source_indices_sha256"] for name, value in data_files.items()},
                     "validation_labels_sha256": digest(labels), "neighbors": graph["neighbors"],
                     "capacity": graph["capacity"], "deployment_root": graph["root"],
                     "quantity_assignment_seed": graph["quantity_assignment_seed"],
                     "capacity_assignment_seed": graph["capacity_assignment_seed"],
                     "local_sample_stream_sha256": digest(sample_streams)}
    if environment.get("initial_state_sha256"):
        pair_identity["initial_state_sha256"] = environment["initial_state_sha256"]
    run.update(status="complete", comparison_group=digest(compatibility)[:16], compatibility=compatibility,
               pair_identity=pair_identity, pair_identity_sha256=digest(pair_identity),
               best_primary=best_row[primary_metric], final_primary=final_row[primary_metric],
               best_round=best_row["round"], best_accuracy=best_row["accuracy"], final_accuracy=final_row["accuracy"],
               best_f1=best_row["f1"], final_f1=final_row["f1"], production_bytes=production_bytes,
               total_probe_examples=sum(row["probe_examples"] for row in run["round_records"]),
               elapsed_seconds=summary["elapsed_seconds"],
               capacity_scope="uniform-rank reference exceeds some declared client caps" if config["arm"] in {"declora16", "product16_sample"} else "training ranks within declared heterogeneous caps",
               input_artifacts_sha256={name: file_digest(folder / name) for name in ["config.json", *required, "summary.json", "source_manifest.json"]})
    return run


def summarize_groups(runs):
    grouped = defaultdict(list)
    for run in runs:
        if run["status"] == "complete":
            grouped[run["comparison_group"]].append(run)
    groups = []
    for group_id, members in sorted(grouped.items()):
        by_arm = defaultdict(dict)
        for run in members:
            if run["seed"] in by_arm[run["arm"]]:
                raise ValueError(f"duplicate complete arm/seed in group {group_id}: {run['arm']}/{run['seed']}; choose explicit run directories")
            by_arm[run["arm"]][run["seed"]] = run
        aggregate = []
        for arm, seed_runs in sorted(by_arm.items()):
            aggregate.append({"arm": arm, "label": LABELS.get(arm, arm),
                              "best_primary": stats({seed: run["best_primary"] for seed, run in sorted(seed_runs.items())}),
                              "final_primary": stats({seed: run["final_primary"] for seed, run in sorted(seed_runs.items())}),
                              "training_bytes": stats({seed: run["training_bytes"] for seed, run in sorted(seed_runs.items())}),
                              "production_bytes": stats({seed: run["production_bytes"] for seed, run in sorted(seed_runs.items())}),
                              "best_and_final_checkpoint_verified_seeds": [seed for seed, run in sorted(seed_runs.items()) if set(run["checkpoint_verification"].get("verified_endpoints", [])) == {"best", "final"}],
                              "final_checkpoint_verified_seeds": [seed for seed, run in sorted(seed_runs.items()) if "final" in run["checkpoint_verification"].get("verified_endpoints", [])]})
        effects = {}
        for name, coefficients in EFFECTS.items():
            available = {arm: sorted(by_arm.get(arm, {})) for arm in coefficients}
            paired = sorted(set.intersection(*(set(values) for values in available.values())))
            result = {"coefficients": coefficients, "available_seeds": available,
                      "matched_seeds": [], "pairing_failures": [], "best_primary_pp": None, "final_primary_pp": None}
            accepted = []
            for seed in paired:
                signatures = {by_arm[arm][seed]["pair_identity_sha256"] for arm in coefficients}
                if len(signatures) != 1:
                    result["pairing_failures"].append({"seed": seed, "reason": "different shards, graph/caps, initialization fingerprint or realized local sample stream"})
                else:
                    accepted.append(seed)
            result["matched_seeds"] = accepted
            if accepted:
                for endpoint in ("best_primary", "final_primary"):
                    values = {seed: sum(coefficient * by_arm[arm][seed][endpoint] for arm, coefficient in coefficients.items()) for seed in accepted}
                    result[endpoint + "_pp"] = stats(values)
                result["status"] = "paired_exploratory_estimate"
            else:
                result["status"] = "no_valid_matched_seed_comparison"
            effects[name] = result
        groups.append({"id": group_id, "compatibility": members[0]["compatibility"], "n_runs": len(members),
                       "seeds": sorted({run["seed"] for run in members}), "aggregate": aggregate,
                       "paired_effects": effects, "figures": [],
                       "interpretation": "exploratory validation comparison; no hidden test, parity test, or direct comparison to printed paper scores"})
    return groups


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value
                          for key, value in row.items()} for row in rows)


def write_tables(output, runs):
    run_fields = ("path", "comparison_group", "status", "classification", "track", "task", "arm", "seed", "primary_metric",
                  "n_train", "n_validation", "n_clients", "best_round", "best_primary", "final_primary", "best_accuracy", "final_accuracy", "best_f1", "final_f1",
                  "total_steps", "total_examples", "total_probe_examples", "training_bytes", "evaluation_assembly_bytes", "production_bytes", "elapsed_seconds", "source_sha256", "split_sha256", "capacity_scope", "warnings")
    per_run = [{**{key: run.get(key) for key in run_fields},
                "rounds_recorded": len(run["round_records"]),
                "checkpoint_verification": run.get("checkpoint_verification", {}).get("status"),
                "checkpoint_verification_kind": run.get("checkpoint_verification", {}).get("kind"),
                "checkpoint_verified_endpoints": run.get("checkpoint_verification", {}).get("verified_endpoints", []),
                "checkpoint_files_present": run.get("checkpoint_files_present"),
                "error": run.get("error")} for run in runs]
    per_round = []
    for run in runs:
        identity = {key: run.get(key) for key in ("path", "comparison_group", "status", "classification", "track", "task", "arm", "seed")}
        per_round.extend({**identity, **row} for row in run["round_records"])
    write_csv(output / "per_run.csv", per_run)
    write_csv(output / "per_round.csv", per_round)


def plot_groups(output, groups, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    for group in groups:
        members = [run for run in runs if run.get("comparison_group") == group["id"] and run["status"] == "complete"]
        context = group["compatibility"]
        title = f"{context['config']['task'].upper()} · {context['config']['partition']} · {context['classification']}"
        by_arm = defaultdict(list)
        for run in members:
            by_arm[run["arm"]].append(run)
        figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
        for arm, values in sorted(by_arm.items()):
            accuracies = np.array([[row["accuracy"] for row in run["round_records"]] for run in values])
            bytes_by_seed = np.array([[row["cumulative_training_bytes"] / 1e9 for row in run["round_records"]] for run in values])
            rounds = np.arange(1, accuracies.shape[1] + 1)
            mean, color = accuracies.mean(0), COLORS.get(arm, "#555555")
            axes[0].plot(rounds, mean, color=color, label=f"{LABELS.get(arm, arm)} (n={len(values)})")
            if len(values) > 1:
                sd = accuracies.std(0, ddof=1)
                axes[0].fill_between(rounds, mean - sd, mean + sd, color=color, alpha=.13)
            # Plot observed seed trajectories rather than interpolate unmatched
            # communication budgets into a fabricated pointwise comparison.
            for index, (x, y) in enumerate(zip(bytes_by_seed, accuracies)):
                axes[1].plot(x, y, color=color, alpha=.65, lw=1.2,
                             label=LABELS.get(arm, arm) if index == 0 else None)
        axes[0].set(xlabel="Communication round", ylabel="Official validation accuracy (%)")
        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[1].set(xlabel="Cumulative training traffic (GB, decimal)", ylabel="Official validation accuracy (%)")
        for axis in axes:
            axis.grid(alpha=.2)
        axes[0].legend(fontsize=7, loc="best")
        figure.suptitle(title)
        scope = " Smoke: execution check only; no efficacy inference." if context["classification"] == "smoke" else ""
        figure.text(.5, .01, "Left: mean ± sample SD when n≥2. Right: observed seed trajectories. Training bytes exclude evaluation assemblies and setup." + scope, ha="center", fontsize=7)
        figure.tight_layout(rect=(0, .045, 1, .94))
        stem = f"{group['id']}_accuracy"
        for suffix in ("pdf", "png"):
            path = figure_dir / f"{stem}.{suffix}"
            figure.savefig(path, dpi=180)
            group["figures"].append(str(path.relative_to(output)))
        plt.close(figure)
        for arm, values in sorted(by_arm.items()):
            fig, axis = plt.subplots(figsize=(7.2, 4.1))
            if len(values) == 1:
                run = values[0]
                traces = np.asarray([row["ranks"] for row in run["round_records"]])
                group_max = max(row["max_rank"] for member in members for row in member["round_records"])
                mesh = axis.imshow(traces.T, aspect="auto", origin="upper", interpolation="nearest",
                                   cmap="viridis", vmin=1, vmax=max(2, group_max),
                                   extent=(.5, len(traces) + .5, traces.shape[1] - .5, -.5))
                axis.set_yticks(np.arange(traces.shape[1]))
                axis.xaxis.set_major_locator(MaxNLocator(integer=True))
                fig.colorbar(mesh, ax=axis, label="Adapter rank").locator = MaxNLocator(integer=True)
                if traces.shape[0] <= 8:
                    for round_index in range(traces.shape[0]):
                        for cid in range(traces.shape[1]):
                            rank = traces[round_index, cid]
                            axis.text(round_index + 1, cid, str(rank), ha="center", va="center", fontsize=7,
                                      color="white" if rank < .6 * group_max else "black")
                detail = f"{context['config']['task'].upper()} · {context['classification']} · seed {run['seed']}; individual peers"
            else:
                means = np.asarray([[row["mean_rank"] for row in run["round_records"]] for run in values])
                x = np.arange(1, means.shape[1] + 1)
                axis.plot(x, means.mean(0), color=COLORS.get(arm, "#555555"))
                axis.fill_between(x, means.mean(0) - means.std(0, ddof=1), means.mean(0) + means.std(0, ddof=1), alpha=.2)
                detail = f"{context['config']['task'].upper()} · {context['classification']} · mean client rank ± seed SD, n={len(values)}"
            axis.set(xlabel="Communication round", ylabel="Peer" if len(values) == 1 else "Mean client rank", title=f"{LABELS.get(arm, arm)}\n{detail}")
            if len(values) > 1:
                axis.grid(alpha=.2)
            fig.tight_layout()
            for suffix in ("pdf", "png"):
                path = figure_dir / f"{group['id']}_ranks_{arm}.{suffix}"
                fig.savefig(path, dpi=180, bbox_inches="tight")
                group["figures"].append(str(path.relative_to(output)))
            plt.close(fig)


def formatted(stat):
    if stat["sample_sd"] is None:
        return f"{stat['mean']:.3f} (one seed)"
    return f"{stat['mean']:.3f} ± {stat['sample_sd']:.3f}"


def markdown(summary):
    lines = ["# Quantity-skew decentralized LoRA results", "",
             "Scores use the official labeled GLUE validation split. SST-2 best validation accuracy is primary; final accuracy is secondary. These are exploratory comparisons of our independent implementations, not hidden-test results or wins over printed paper numbers.", "",
             "Smoke runs are execution checks only. Their scores and paired arithmetic do not establish efficacy or failure of the proposed method.", "",
             f"Runs: {summary['n_complete']} complete, {summary['n_incomplete']} incomplete, {summary['n_invalid']} invalid. In-progress runs are inventoried only and excluded from every aggregate, paired effect, and figure; any partial round rows retain incomplete status in per_round.csv. Complete runs enter only source/model/data/budget-compatible groups. Smoke and full configured budgets, and equal-size anchors and quantity-skew extensions, remain separate.", "",
             "Metrics were independently recomputed from saved predictions and validation labels. The standalone checkpoint audit separately verifies best/final inference, source/data integrity, and exact raw prediction matches; its documented model/install-helper dependency is shared. Archived checkpoint binaries may be absent when their exact hashes are bound to that audit. All intervals below are unadjusted Student-t intervals of seed-paired differences; small samples are exploratory, and nonsignificance does not establish parity.", ""]
    for group in summary["groups"]:
        context, config = group["compatibility"], group["compatibility"]["config"]
        lines += [f"## Group {group['id']}", "",
                  f"{config['task'].upper()} · {config['partition']} · {context['classification']}; {config['rounds']} rounds, batch {config['batch_size']}, local steps {config.get('local_steps', 0)}, local epochs {config.get('local_epochs', 1)}. Total local updates {context['total_training_steps']:,}; training-example exposures {context['total_training_examples']:,}. Source `{context['source_sha256']}`.", "",
                  "A full configured budget means all training data are available and at least 20 rounds run; it does not imply an exact reproduction of the publication's budget. Rank-16 reference arms exceed weaker clients' rank caps. The assembly root stores routed adapter records; neither routing nor raw-data locality establishes privacy.", "",
                  f"Primary metric: best validation {context['primary_metric']} (%). Mean ± sample SD across seeds; n=1 has no estimated between-seed uncertainty.", "",
                  "| Method | Seeds | Best primary | Final (secondary) | Training GB | Best + final checkpoint audits |",
                  "|---|---:|---:|---:|---:|---:|"]
        for row in group["aggregate"]:
            lines.append(f"| {row['label']} | {row['best_primary']['n']} | {formatted(row['best_primary'])} | {formatted(row['final_primary'])} | {row['training_bytes']['mean']/1e9:.3f} | {len(row['best_and_final_checkpoint_verified_seeds'])}/{row['best_primary']['n']} |")
        lines += ["", "Paired effects use each seed's own best endpoint, even when best rounds differ. Final-round effects are also recorded in JSON. Main effects average the two within-factor contrasts; the interaction is the difference of those differences.", "",
                  "| Paired effect | Matched seeds | Best endpoint difference (pp) | Exploratory 95% t interval |",
                  "|---|---:|---:|---:|"]
        for name, effect in group["paired_effects"].items():
            stat = effect["best_primary_pp"]
            if stat is None:
                lines.append(f"| {name} | 0 | unavailable | unavailable |")
                continue
            interval = "unavailable (one seed)" if stat["ci95"] is None else f"[{stat['ci95'][0]:.3f}, {stat['ci95'][1]:.3f}]"
            lines.append(f"| {name} | {stat['n']} | {formatted(stat)} | {interval} |")
        failures = sum(len(effect["pairing_failures"]) for effect in group["paired_effects"].values())
        if failures:
            lines += ["", f"{failures} effect/seed combinations were excluded because pairing invariants failed; inspect summary.json."]
        lines += [""]
        pngs = [path for path in group["figures"] if path.endswith("_accuracy.png")]
        if pngs:
            lines += [f"![Validation accuracy versus rounds and training traffic]({pngs[0]})", ""]
    flagged = [run for run in summary["runs"] if run.get("warnings") or run.get("error")]
    if flagged:
        lines += ["## Run status and verification gaps", ""]
        for run in flagged:
            detail = "; ".join(([run["error"]] if run.get("error") else []) + run.get("warnings", []))
            lines.append(f"- `{run['path']}` — {run['status']}: {detail}")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true", help="Write audited tables/summary only.")
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] != "gpu003":
        raise RuntimeError("Scientific analysis and plotting are restricted to authorized gpu003")
    folders = run_directories(args.inputs)
    if not folders:
        raise ValueError("no quantity benchmark run directories found")
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    for folder in folders:
        try:
            run = load_run(folder)
        except (ValueError, KeyError, TypeError, OSError, OverflowError) as error:
            run = {"path": str(folder), "status": "invalid", "round_records": [], "warnings": [],
                   "error": f"{type(error).__name__}: {error}"}
        runs.append(run)
    groups = summarize_groups(runs)
    write_tables(args.output, runs)
    if not args.no_plots:
        plot_groups(args.output, groups, runs)
    summary = {"schema_version": 1, "analysis_host": socket.gethostname(), "analysis_script_sha256": file_digest(__file__),
               "n_complete": sum(run["status"] == "complete" for run in runs),
               "n_incomplete": sum(run["status"] == "incomplete" for run in runs),
               "n_invalid": sum(run["status"] == "invalid" for run in runs),
               "metric_scope": "official labeled validation; best SST-2 accuracy primary, final secondary; no hidden test",
               "uncertainty_scope": "sample SD and unadjusted seed-paired Student-t 95% intervals only for n>=2; exploratory, no equivalence or superiority certification",
               "groups": groups, "runs": [{key: value for key, value in run.items() if key != "round_records"} for run in runs]}
    write_json(args.output / "summary.json", summary)
    (args.output / "SUMMARY.md").write_text(markdown(summary))
    print(json.dumps({key: summary[key] for key in ("n_complete", "n_incomplete", "n_invalid")}, sort_keys=True))
    return 1 if summary["n_invalid"] else 0


if __name__ == "__main__":
    sys.exit(main())
