#!/usr/bin/env python3
"""Independent final quantity-study artifact audit; execute only on gpu003.

This script imports no experiment, transport, evaluator or summary code. It
rechecks saved evidence and calculations, not model inference. Existing exact
inference audits are bound to current checkpoint bytes and source hashes.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import socket


ARMS = ("declora16", "declora4", "product16_sample", "fixed_uniform",
        "fixed_sample", "adaptive_uniform", "adaptive_sample")
SEEDS = (42, 43, 44, 45, 46)
SOURCE = "daced6fc0052f4aab6069aea78ce1468bd5c039b76dedb71080155b0276b8d61"
MODEL = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
DATA = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def close(left, right, label):
    require(math.isclose(left, right, rel_tol=0, abs_tol=1e-10), label)


def metric(record, truth):
    predictions = record.get("predictions")
    require(predictions is not None and len(predictions) == len(truth), "prediction length")
    require(all(type(v) is int and v in (0, 1) for v in predictions), "binary predictions")
    correct = sum(p == y for p, y in zip(predictions, truth))
    tp = sum(p == y == 1 for p, y in zip(predictions, truth))
    fp = sum(p == 1 and y == 0 for p, y in zip(predictions, truth))
    fn = sum(p == 0 and y == 1 for p, y in zip(predictions, truth))
    accuracy = 100 * correct / len(truth)
    f1 = 200 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
    raw = b"".join(p.to_bytes(8, "little", signed=True) for p in predictions)
    result = {"correct": correct, "n": len(truth), "accuracy": accuracy,
              "f1": f1, "predictions_sha256": hashlib.sha256(raw).hexdigest()}
    compare_metric(record, result)
    return result


def compare_metric(record, expected):
    for key in ("correct", "n", "predictions_sha256"):
        require(record[key] == expected[key], "metric identity: " + key)
    for key in ("accuracy", "f1"):
        close(record[key], expected[key], "metric arithmetic: " + key)


def transport(ledger, graph, kind):
    rows = ledger["transfers"]
    require(len(rows) == ledger["messages"], "message total")
    require(len(rows) == (20 if kind == "gossip" else 9), "declared transfer count")
    for row in rows:
        sender, receiver = row["sender"], row["receiver"]
        require(receiver in graph["neighbors"][str(sender)], "undeclared edge")
        require(row["phase"] == kind, "unexpected transmission phase")
        require(row["tensor_bytes"] + row["metadata_bytes"] == row["payload_bytes"], "wire byte sum")
        require(re.fullmatch(r"[0-9a-f]{64}", row["payload_sha256"]) is not None, "payload hash format")
        require(row["record_count"] == len(row["source_client_ids"]) == len(row["source_weights"]), "record count")
        if kind == "gossip":
            require(row["source_client_ids"] == [sender], "gossip source")
            close(row["source_weights"][0], graph["matrix"][receiver][sender], "gossip coefficient")
        else:
            require(ledger["tree_parent"][str(sender)] == receiver, "gather tree")
            for client, weight in zip(row["source_client_ids"], row["source_weights"]):
                close(weight, graph["stationary_weights"][client], "gather coefficient")
    for field, rowfield in (("bytes", "payload_bytes"), ("tensor_bytes", "tensor_bytes"), ("metadata_bytes", "metadata_bytes")):
        require(ledger[field] == sum(r[rowfield] for r in rows), "ledger totals")
    if kind == "gather":
        require(ledger["root_id"] == graph["root"] and ledger["disseminated"] is False, "deployment endpoint")
        require(ledger["dissemination_messages"] == 0, "unexpected deployment broadcast")


def stats(values):
    import numpy as np
    from scipy.stats import t
    values = {str(k): float(v) for k, v in values.items()}
    a = np.asarray(list(values.values()))
    result = {"n": len(a), "per_seed": values, "mean": float(a.mean()), "sample_sd": None, "ci95_t": None}
    if len(a) > 1:
        sd = float(a.std(ddof=1))
        half = float(t.ppf(.975, len(a) - 1) * sd / math.sqrt(len(a)))
        result.update(sample_sd=sd, ci95_t=[result["mean"] - half, result["mean"] + half])
    return result


def main():
    require(socket.gethostname().split(".")[0] == "gpu003", "Scientific execution is restricted to gpu003")
    import numpy as np
    import pyarrow.parquet as parquet
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base, output = args.base.expanduser(), args.output.expanduser()
    campaign = read(base / "campaign-full-v1/status.json")
    expected = {(arm, seed, "quantity") for arm in ARMS for seed in SEEDS} | {("declora16", 42, "equal")}
    require(campaign["status"] == "complete" and campaign["progress"]["verified_jobs"] == 36, "campaign completion")
    require(len(campaign["plan"]) == 36 and len(campaign["jobs"]) == 36, "campaign plan count")
    actual = {(p["arm"], p["seed"], p["partition"]) for p in campaign["plan"]}
    require(actual == expected, "registered method/seed coverage")
    assets = base / "assets"
    tables, data_hashes = {}, {}
    for split in ("train", "validation"):
        path = assets / "glue/sst2" / (split + "-00000-of-00001.parquet")
        tables[split] = parquet.read_table(path, columns=["label"]).to_pydict()["label"]
        data_hashes[split] = sha(path)
    require(len(tables["train"]) == 67349 and len(tables["validation"]) == 872, "canonical data sizes")
    model_hashes = {p.name: sha(p) for p in (assets / "roberta-base").iterdir() if p.is_file()}
    results, pair_ids, configurations, environment_ids = [], {}, {}, set()
    for item in sorted(campaign["plan"], key=lambda p: p["id"]):
        name, arm, seed, partition = item["id"], item["arm"], item["seed"], item["partition"]
        run = base / "runs" / name
        require(campaign["jobs"][name]["status"] == "complete", name + " job state")
        config, summary, audit, env, split, graph, source = [read(run / f) for f in
            ("config.json", "summary.json", "independent_checkpoint_audit.json", "environment.json", "split.json", "graph.json", "source_manifest.json")]
        require(config["arm"] == arm and config["seed"] == seed and config["partition"] == partition, "config identity")
        for key, value in {"task": "sst2", "rounds": 20, "local_epochs": 1, "local_steps": 211 if partition == "quantity" else 0,
                           "batch_size": 32, "max_length": 128, "lr": .001, "alpha": 16., "max_grad_norm": 1., "max_train": 0,
                           "verify_only": False}.items():
            require(config[key] == value, "registered configuration: " + key)
        stable_config = {k: v for k, v in config.items() if k not in ("arm", "seed", "output")}
        if partition in configurations:
            require(stable_config == configurations[partition], "cross-arm hyperparameters")
        configurations[partition] = stable_config
        require(summary["status"] == "complete" and summary["rounds"] == 20 and summary["primary_metric"] == "accuracy", "summary completion")
        source_digest = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
        require(source_digest == summary["source_sha256"] == env["source_sha256"] == SOURCE, "source identity")
        for path, expected_sha in source.items():
            require(sha(run / "source" / path) == expected_sha, "archived source hash: " + path)
        require(env["model_revision"] == MODEL and env["dataset_revision"] == DATA, "asset revisions")
        require(env["model_files"] == model_hashes, "model asset hashes")
        environment_ids.add(digest({k: env[k] for k in ("model_files", "model_revision", "dataset_revision", "precision", "optimizer_policy", "adapter_policy")}))
        memberships = {str(i): sorted(split["clients"][str(i)]["indices"]) for i in range(10)}
        require(sorted(v for ids in memberships.values() for v in ids) == list(range(67349)), "disjoint exhaustive shards")
        require(digest(memberships) == split["partition_sha256"], "partition digest")
        counts = split["client_counts"]
        require(split["validation_labels"] == tables["validation"], "validation targets")
        totals = Counter(tables["train"])
        for i in range(10):
            ids = memberships[str(i)]
            require(len(ids) == counts[i], "shard count")
            observed = Counter(tables["train"][j] for j in ids)
            require([observed[c] for c in (0, 1)] == split["clients"][str(i)]["class_counts"], "class counts")
            for c in (0, 1):
                quota = counts[i] * totals[c] / 67349
                require(observed[c] in (math.floor(quota), math.ceil(quota)), "proportional class quota")
        for label in ("train", "validation"):
            declared = split["data_files"][label]
            require(declared["file_sha256"] == data_hashes[label], "data file hash")
            require(declared["rows_used"] == declared["rows_in_file"] == len(tables[label]), "data row counts")
            require(declared["source_indices"] == list(range(len(tables[label]))), "full source-row mapping")
        for i in range(10):
            require(graph["neighbors"][str(i)] == sorted(((i-1) % 10, (i+1) % 10)), "ten-peer ring")
        caps = graph["capacity"]
        require(sorted(caps) == [4, 4, 4, 4, 8, 8, 8, 16, 16, 16], "capacity schedule")
        require(graph["root"] == caps.index(16), "preselected capable root")
        weights = np.array(counts) / sum(counts) if arm.endswith("sample") else np.ones(10) / 10
        matrix = np.asarray(graph["matrix"])
        expected_matrix = np.zeros((10, 10))
        for i in range(10):
            for j in graph["neighbors"][str(i)]:
                expected_matrix[i, j] = min(1, weights[j] / weights[i]) / 3
            expected_matrix[i, i] = 1 - expected_matrix[i].sum()
        require(np.allclose(graph["stationary_weights"], weights, atol=1e-12, rtol=0), "objective weights")
        require(np.allclose(matrix, expected_matrix, atol=1e-12, rtol=0), "Metropolis matrix")
        require(graph["setup_bytes"] == 320, "modeled setup allowance")
        rounds = [json.loads(line) for line in (run / "rounds.jsonl").read_text().splitlines()]
        require([r["round"] for r in rounds] == list(range(1, 21)), "all twenty rounds")
        streams, metrics = [], []
        training_bytes = assembly_bytes = total_steps = total_examples = probes = 0
        for row in rounds:
            metrics.append(metric(row["validation"], tables["validation"]))
            require(set(row["local"]) == {str(i) for i in range(10)}, "all local peer records")
            streams.append({cid: {k: local[k] for k in ("steps", "examples", "sample_order_sha256")} for cid, local in row["local"].items()})
            for cid, local in row["local"].items():
                expected_steps = 211 if partition == "quantity" else math.ceil(counts[int(cid)] / 32)
                expected_examples = 6752 if partition == "quantity" else counts[int(cid)]
                require(local["steps"] == expected_steps and local["examples"] == expected_examples, "per-peer budget")
                total_steps += local["steps"]
                total_examples += local["examples"]
            require(len(row["ranks"]) == 10, "rank trace")
            if arm in ("declora16", "product16_sample"):
                require(row["ranks"] == [16] * 10, "rank16 baseline")
            elif arm == "declora4":
                require(row["ranks"] == [4] * 10, "rank4 baseline")
            elif arm.startswith("fixed"):
                require(row["ranks"] == caps, "fixed capacity ranks")
            else:
                require(all(c / 2 <= r <= c for r, c in zip(row["ranks"], caps)), "adaptive rank bounds")
                require(set(row["probes"]) == {str(i) for i in range(10)}, "adaptive probe coverage")
                for cid, probe in row["probes"].items():
                    require(probe["before"]["probe_rank"] == caps[int(cid)] and probe["before"]["probe_examples"] == 32, "capability probe")
            probes += sum(p["before"]["probe_examples"] * 2 for p in row["probes"].values())
            transport(row["gossip"], graph, "gossip")
            transport(row["evaluation_assembly"], graph, "gather")
            require(row["evaluation_assembly"]["target_rank"] == (4 if arm == "declora4" else 16), "deployment rank")
            training_bytes += row["gossip"]["bytes"]
            assembly_bytes += row["evaluation_assembly"]["bytes"]
            require(row["cumulative_training_bytes"] == training_bytes and row["cumulative_evaluation_assembly_bytes"] == assembly_bytes, "cumulative bytes")
        require(summary["total_steps"] == total_steps == 42200, "total optimizer steps")
        require(summary["total_examples"] == total_examples == (1350400 if partition == "quantity" else 1346980), "total exposures")
        require(summary["total_probe_examples"] == probes == (12800 if arm.startswith("adaptive") else 0), "probe exposures")
        require(summary["training_bytes"] == training_bytes and summary["evaluation_assembly_bytes"] == assembly_bytes, "summary bytes")
        production = 320 + training_bytes + rounds[-1]["evaluation_assembly"]["bytes"]
        require(summary["production_bytes_setup_training_final_assembly"] == production, "production bytes")
        require(summary["all_experiment_bytes_including_evaluations"] == 320 + training_bytes + assembly_bytes, "experiment bytes")
        best_index = max(range(20), key=lambda i: metrics[i]["accuracy"])
        require(summary["best"]["round"] == best_index + 1, "first maximum selection")
        require(audit["status"] == "passed" and set(audit["checkpoints"]) == {"best", "final"}, "both inference audits")
        require(audit["round_metric_records_verified"] == 20 and audit["host"].split(".")[0] == "gpu003", "audit execution")
        checks = audit["artifact_checks"]
        require(checks["source_sha256"] == SOURCE and checks["partition_sha256"] == split["partition_sha256"], "audit source/split binding")
        require(checks["shards_disjoint_exhaustive"] and checks["quantity_only_class_quotas_verified"], "audit data invariants")
        require(checks["partition_examples"] == 67349 and checks["validation_rows"] == 872, "audit data sizes")
        require(audit["evaluator_sha256"] == source["project-3-hierarchical-gossip/experiments/verify_quantity_checkpoint.py"], "audit evaluator source")
        require(audit["model_helper_sha256"] == source["project-3-hierarchical-gossip/src/models/roberta_lora.py"], "audit model helper")
        checkpoint_hashes = {}
        for endpoint, index in (("best", best_index), ("final", 19)):
            record = audit["checkpoints"][endpoint]
            require(record["round"] == index + 1 and record["exact_raw_prediction_match"] is True, "audited checkpoint identity")
            compare_metric(record["metrics"], metrics[index])
            compare_metric(summary["best"]["validation"] if endpoint == "best" else summary["final"], metrics[index])
            checkpoint_hashes[endpoint] = sha(run / (endpoint + ".pt"))
            require(checkpoint_hashes[endpoint] == record["checkpoint_sha256"] == summary[endpoint + "_checkpoint_sha256"], "current checkpoint hash")
        checkpoint_hashes["peers_final"] = sha(run / "peers_final.pt")
        require(checkpoint_hashes["peers_final"] == summary["peer_checkpoint_sha256"], "peer state archive hash")
        pair = {"partition": split["partition_sha256"], "labels": split["label_sha256"], "capacities": caps,
                "root": graph["root"], "neighbors": graph["neighbors"], "streams": digest(streams)}
        if partition == "quantity":
            if seed in pair_ids:
                require(pair_ids[seed] == pair, "cross-arm shard/capacity/sample-stream pairing")
            pair_ids[seed] = pair
        results.append({"run": name, "arm": arm, "seed": seed, "partition": partition,
            "best_accuracy": metrics[best_index]["accuracy"], "best_round": best_index + 1,
            "final_accuracy": metrics[-1]["accuracy"], "training_GB": training_bytes / 1e9,
            "production_GB": production / 1e9, "evaluation_assembly_GB": assembly_bytes / 1e9,
            "mean_persistent_rank": sum(sum(r["ranks"]) for r in rounds) / 200,
            "peak_process_cuda_GB": max(r["peak_process_cuda_allocated_bytes"] for r in rounds) / 1e9,
            "probe_examples": probes, "checkpoint_hashes": checkpoint_hashes,
            "source_files_verified": len(source), "pair_identity_sha256": digest(pair),
            "input_hashes": {f: sha(run / f) for f in ("summary.json", "independent_checkpoint_audit.json", "config.json", "environment.json", "source_manifest.json", "split.json", "graph.json", "rounds.jsonl")}})
        print("verified " + name, flush=True)
    require(len(environment_ids) == 1, "common model/optimizer environment")
    by_arm = {arm: {r["seed"]: r for r in results if r["partition"] == "quantity" and r["arm"] == arm} for arm in ARMS}
    aggregates = {arm: {key: stats({seed: by_arm[arm][seed][key] for seed in SEEDS}) for key in
        ("best_accuracy", "final_accuracy", "training_GB", "production_GB", "evaluation_assembly_GB", "mean_persistent_rank", "peak_process_cuda_GB", "probe_examples")} for arm in ARMS}
    contrasts = {
        "adaptive_sample_minus_declora16": {"adaptive_sample": 1, "declora16": -1},
        "adaptive_sample_minus_declora4": {"adaptive_sample": 1, "declora4": -1},
        "adaptive_sample_minus_product16_sample": {"adaptive_sample": 1, "product16_sample": -1},
        "adaptation_at_uniform_weights": {"adaptive_uniform": 1, "fixed_uniform": -1},
        "adaptation_at_sample_weights": {"adaptive_sample": 1, "fixed_sample": -1},
        "sample_weighting_at_fixed_ranks": {"fixed_sample": 1, "fixed_uniform": -1},
        "sample_weighting_at_adaptive_ranks": {"adaptive_sample": 1, "adaptive_uniform": -1},
        "adaptation_main_effect": {"adaptive_sample": .5, "adaptive_uniform": .5, "fixed_sample": -.5, "fixed_uniform": -.5},
        "sample_weighting_main_effect": {"adaptive_sample": .5, "fixed_sample": .5, "adaptive_uniform": -.5, "fixed_uniform": -.5},
        "adaptation_by_weighting_interaction": {"adaptive_sample": 1, "fixed_uniform": 1, "adaptive_uniform": -1, "fixed_sample": -1}}
    effects = {name: {endpoint: stats({seed: sum(coef * by_arm[arm][seed][endpoint] for arm, coef in coefficients.items()) for seed in SEEDS})
                        for endpoint in ("best_accuracy", "final_accuracy")} for name, coefficients in contrasts.items()}
    traffic = {arm: stats({seed: 100 * (1 - by_arm["adaptive_sample"][seed]["training_GB"] / by_arm[arm][seed]["training_GB"]) for seed in SEEDS})
               for arm in ("declora16", "declora4", "fixed_sample")}
    result = {"status": "passed", "recorded_utc": datetime.now(timezone.utc).isoformat(), "host": socket.gethostname(),
        "audit_script_sha256": sha(Path(__file__)), "campaign_sha256": sha(base / "campaign-full-v1/status.json"),
        "campaign_completed_utc": campaign["updated_at"], "full_runs_verified": 36, "quantity_runs_verified": 35,
        "checkpoint_inference_audits_bound": 72, "current_checkpoint_files_hashed": 108,
        "round_prediction_records_recomputed": 720, "source_sha256": SOURCE, "model_revision": MODEL, "dataset_revision": DATA,
        "model_asset_hashes": model_hashes, "dataset_hashes": data_hashes, "seeds": list(SEEDS),
        "independence_scope": "No training/transport/checkpoint-evaluator/summary imports; independent artifact hashes, parquet labels, quotas, pairing, ledger/metric arithmetic and statistics. Existing exact inference audits are evidence, not rerun here. They share the Hugging Face model and LoRA install helper.",
        "limitations": ["Saved message SHA256 strings and ledger arithmetic checked; raw wire buffers are not retained, so their payloads are not reconstructed.",
            "Initial model state hashes were not recorded. Common initialization is supported by pinned same-seed code and model unit tests, not a per-run initial-state artifact comparison.",
            "Confidence intervals are unadjusted exploratory Student-t intervals across five seeds, not across tasks or fresh evaluation samples; no predeclared noninferiority margin exists.",
            "Best labeled-validation accuracy is primary; final-round accuracy is secondary; hidden test was not used.",
            "Modeled count setup is 320 bytes. Production traffic excludes base/initial/topology provisioning and network framing. Whole-process CUDA peak is not independent-client memory.",
            "Single-process peer simulation with visible updates and a gathering root; no enforced privacy guarantee or real multi-machine deployment."],
        "aggregates": aggregates, "paired_effects_pp": effects, "adaptive_training_traffic_reduction_percent": traffic,
        "equal_anchor": next(r for r in results if r["partition"] == "equal"), "runs": results}
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
    lines = ["# Independent final quantity-study audit", "", "All 36 full runs pass the artifact audit: 35 matched quantity runs and one separate equal-size anchor. All 72 recorded best/final inference audits are bound to current checkpoint bytes; 36 peer-state archives are also hashed. All 720 round metrics are recomputed from saved predictions and canonical validation labels.", "", "This script independently checks artifact consistency and calculations. It verifies the recorded inference-audit evidence against the saved models and predictions; it does not rerun model inference or import the experiment/summary implementations.", "", "## Completed five-seed results", "", "Accuracy is percent, mean ± sample SD across seeds 42–46. Best official SST-2 validation accuracy is primary; final accuracy is secondary.", "", "| Arm | Best validation | Final validation | Training GB | Production GB | Mean persistent rank |", "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        a = aggregates[arm]
        fmt = lambda key: f"{a[key]['mean']:.3f} ± {a[key]['sample_sd']:.3f}"
        lines.append(f"| {arm} | {fmt('best_accuracy')} | {fmt('final_accuracy')} | {a['training_GB']['mean']:.3f} | {a['production_GB']['mean']:.3f} | {a['mean_persistent_rank']['mean']:.3f} |")
    lines += ["", "## Paired comparisons and factorial effects", "", "Differences are percentage points. Intervals are exploratory, unadjusted 95% Student-t intervals across the five paired seeds.", "", "| Contrast | Best difference ± SD | Best 95% interval | Final difference ± SD | Final 95% interval |", "|---|---:|---:|---:|---:|"]
    for name, effect in effects.items():
        b, f = effect["best_accuracy"], effect["final_accuracy"]
        lines.append(f"| {name} | {b['mean']:+.3f} ± {b['sample_sd']:.3f} | [{b['ci95_t'][0]:+.3f}, {b['ci95_t'][1]:+.3f}] | {f['mean']:+.3f} ± {f['sample_sd']:.3f} | [{f['ci95_t'][0]:+.3f}, {f['ci95_t'][1]:+.3f}] |")
    lines += ["", "## Interpretation", "", "The planned comparison is complete. The proposed adaptive/sample method has lower mean best and final validation accuracy than rank 16, feasible rank 4, and fixed/sample controls. It demonstrates reduced training traffic against rank 16 and fixed/sample, while using more traffic than rank 4. These findings establish a measured accuracy/traffic tradeoff; they do not demonstrate superiority or accuracy parity. A best-endpoint interval crossing zero does not establish equivalence. Neither adaptive ranks nor sample weighting shows an accuracy advantage in the factorial average. The results are scoped to this reconstruction, task, quantities, capacity schedule and budget.", "", "Adaptive training-traffic reductions relative to each comparator:", ""]
    for arm, value in traffic.items():
        lines.append(f"- {arm}: {value['mean']:.3f}% (negative means more adaptive traffic).")
    lines += ["", "## Verification limits", ""] + ["- " + value for value in result["limitations"]]
    lines += ["", "The original disjoint-domain negative study is preserved separately. Final paper integration and PDF validation are delivery tasks beyond this numerical audit.", ""]
    (output / "AUDIT.md").write_text("\n".join(lines), encoding="utf8")
    print(json.dumps({"status": "passed", "output": str(output), "full_runs": 36, "inference_audits": 72}), flush=True)


if __name__ == "__main__":
    main()
