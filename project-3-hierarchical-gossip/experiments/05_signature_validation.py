"""Compare adapter signatures after local-only and flat MH training.

Usage from project-3-hierarchical-gossip (run on the SSH GPU machine):
    python experiments/05_signature_validation.py --output results/signatures \
        --methods local mh --seeds 42 43 44 --stages 2 5 10 20 \
        --ranks 16 --alpha 32 --consensus-rank 16

Remaining options are shared with 04_protocol_benchmark.py. True domain labels
are used to construct the benchmark data split and score clusters, never as
signature inputs or flat-topology inputs. The cluster count is prespecified.
This evaluates an offline discovery signal; it does not deploy a distributed
clustering protocol or account for exchanging signatures.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch

from experiments.feature_cache import load_features, tensor_digest, write_json
from experiments.protocol_benchmark import (
    FeatureClient, build_mixing, evaluate_protocol, initial_parameters,
    make_splits, package_versions, parse_args as parse_benchmark_args, source_provenance,
)
from src.clustering.domain_clustering import extract_lora_features
from src.clustering.signatures import (
    affinity_matrix, cluster_from_affinity, signature_delta_vec, signature_row_norms,
)
from src.federated.runner import DecentralizedRunner


SIGNATURES = ("spectral_baseline", "row_norms", "inverse_delta_l2")
SUMMARY_FIELDS = ["seed", "method", "stage", "signature", "adjusted_rand_index",
                  "normalized_mutual_info", "personalized_accuracy", "consensus_accuracy"]


def score_signatures(states, true_domains, alpha, n_clusters, seed, stage):
    """Cluster adapters before consulting labels; randomize tie-breaking order.

    Client IDs in this testbed are grouped by domain. A shuffled row order avoids
    turning agglomerative ties between uninformative adapters into an apparent
    discovery success simply because consecutive IDs share a true domain.
    """
    client_order = np.random.default_rng(seed + 7919 * stage).permutation(sorted(states)).tolist()
    ordered = {cid: states[cid] for cid in client_order}
    records = []
    for name in SIGNATURES:
        record = {"signature": name, "clustering_client_order": client_order}
        if name == "spectral_baseline":
            features, ids = extract_lora_features(ordered)
            if not np.isfinite(features).all():
                raise ValueError("non-finite spectral features")
            predicted = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(features)
            record["features"] = features.tolist()
            record["clustering"] = "Ward linkage on standardized original A/B/BA spectral features"
        else:
            signature, kind = ((signature_row_norms, "cosine") if name == "row_norms"
                               else (signature_delta_vec, "inv_l2"))
            vectors = [signature(state, alpha) for state in ordered.values()]
            affinity = affinity_matrix(vectors, kind=kind)
            predicted = cluster_from_affinity(affinity, n_clusters=n_clusters)
            ids = client_order
            record["affinity"] = affinity.tolist()
            record["signature_dimension"] = len(vectors[0])
            record["clustering"] = "Average linkage on distance = 1 - affinity"
            if name == "row_norms":
                record["features"] = np.asarray(vectors).tolist()
            else:
                record["signature_sha256"] = hashlib.sha256(np.asarray(vectors).tobytes()).hexdigest()
        assignments = {cid: int(label) for cid, label in zip(ids, predicted)}
        truth = [true_domains[cid] for cid in ids]
        record.update({"assignments": assignments,
                       "adjusted_rand_index": float(adjusted_rand_score(truth, predicted)),
                       "normalized_mutual_info": float(normalized_mutual_info_score(truth, predicted))})
        records.append(record)
    return records


def run_one(config, seed, method, train_cpu, test_cpu, feature_metadata, output):
    started = time.perf_counter()
    config.current_seed = seed
    splits, domains = make_splits(train_cpu["labels"].numpy(), test_cpu["labels"].numpy(), config, seed)
    split_digest = hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()
    write_json(output / f"splits_seed{seed}.json", {"seed": seed, "sha256": split_digest, "clients": splits})
    ranks = {cid: config.ranks[cid % len(config.ranks)] for cid in domains}
    if len(set(ranks.values())) != 1:
        raise ValueError("signature comparison requires a common rank for the original spectral baseline")
    initial = initial_parameters(train_cpu["features"].shape[1], max(ranks.values()), seed)
    initial_digest = tensor_digest(*initial.values())
    device = torch.device(config.device)
    train = {key: value.to(device) for key, value in train_cpu.items()}
    test = {key: value.to(device) for key, value in test_cpu.items()}
    clients = [FeatureClient(cid, domains[cid], initial, ranks[cid], config, train, test,
                             splits[cid], device) for cid in sorted(domains)]
    # build_mixing's seeded ring order is independent of domain memberships.
    mixing = build_mixing(method, domains, config.bridge_every, config.topology, seed=seed)
    runner = DecentralizedRunner(clients, mixing, ranks, config.alpha, error_feedback=False)
    evaluation_indices = torch.tensor(sorted(index for split in splits.values()
                                             for index in split["test_indices"]), device=device)
    consensus_rank = config.consensus_rank or max(ranks.values())
    record = {"schema_version": 1, "status": "running", "seed": seed, "method": method,
              "ranks": ranks, "alpha": config.alpha, "consensus_rank": consensus_rank,
              "initial_state_sha256": initial_digest, "split_sha256": split_digest,
              "true_domains_for_scoring": domains, "n_clusters_prespecified": config.n_clusters,
              "feature_cache_sha256": {key: feature_metadata[key]["sha256"] for key in ("train", "test")},
              "mixing_matrix": mixing(0).tolist(),
              "n_train_samples": sum(len(split["train_indices"]) for split in splits.values()),
              "n_test_samples": len(evaluation_indices), "stages": [], "training_rounds": []}
    path = output / f"seed{seed}_{method}.json"
    write_json(path, record)
    for round_idx in range(max(config.stages)):
        training = [client.train() for client in clients]
        states = [client.get_lora_state() for client in clients]
        if method == "mh":
            new_states, diagnostics = runner.gossip_round(round_idx, states)
            for client, state in zip(clients, new_states):
                client.set_lora_state(state)
        else:
            diagnostics = {"messages": 0, "floats": 0, "mean_tail_mass": 0.0,
                           "max_tail_mass": 0.0, "mean_residual_energy": 0.0,
                           "consensus_distance": runner._consensus_distance(states)}
        record["training_rounds"].append({"round": round_idx + 1,
             "mean_train_loss": float(np.mean([value["loss"] for value in training])), **diagnostics})
        if round_idx + 1 in config.stages:
            stage = round_idx + 1
            states = {client.client_id: client.get_lora_state() for client in clients}
            metrics = evaluate_protocol(clients, runner, initial, test, evaluation_indices,
                                        consensus_rank, config.eval_batch_size)
            signatures = score_signatures(states, domains, config.alpha, config.n_clusters, seed, stage)
            record["stages"].append({"stage": stage, "evaluation": metrics, "signatures": signatures,
                                      "adapter_state_sha256": tensor_digest(*[
                                          value for cid in sorted(states) for layer in sorted(states[cid])
                                          for value in states[cid][layer].values()])})
            for signature in signatures:
                print(f"seed={seed} method={method} stage={stage} signature={signature['signature']} "
                      f"ARI={signature['adjusted_rand_index']:.4f} "
                      f"NMI={signature['normalized_mutual_info']:.4f}", flush=True)
        record["wall_seconds"] = time.perf_counter() - started
        write_json(path, record)
    record["status"] = "complete"
    record["wall_seconds"] = time.perf_counter() - started
    write_json(path, record)
    return record


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stages", nargs="+", type=int, default=[2, 5, 10, 20])
    parser.add_argument("--n-clusters", type=int, default=5)
    own, remaining = parser.parse_known_args(argv)
    if "--methods" not in remaining:
        remaining.extend(["--methods", "local", "mh"])
    config = parse_benchmark_args(remaining)
    if (not own.stages or min(own.stages) < 1
            or own.stages != sorted(set(own.stages))):
        parser.error("stages must be unique, positive and increasing")
    if own.n_clusters < 2 or own.n_clusters > config.n_domains * config.clients_per_domain:
        parser.error("n-clusters must be between 2 and the client count")
    if any(method not in {"local", "mh"} for method in config.methods):
        parser.error("signature validation supports only --methods local mh")
    if config.merge != "delta" or config.error_feedback or config.prepare_only:
        parser.error("signature validation uses delta merging without error feedback; use 04 for preparation")
    if len(set(config.ranks)) != 1:
        parser.error("the spectral baseline needs a homogeneous rank schedule")
    config.stages, config.n_clusters = own.stages, own.n_clusters
    config.rounds = max(own.stages)
    return config


def summarize_gate(records):
    """Descriptive, prespecified stage-10 screen; not a statistical acceptance test."""
    groups = {}
    for record in records:
        for stage in record["stages"]:
            if stage["stage"] == 10:
                for signature in stage["signatures"]:
                    key = (record["method"], signature["signature"])
                    groups.setdefault(key, []).append(signature["adjusted_rand_index"])
    summary = []
    for (method, signature), values in groups.items():
        mean = float(np.mean(values))
        summary.append({"method": method, "signature": signature, "stage": 10,
                        "n_seeds": len(values), "mean_ari": mean,
                        "sample_std_ari": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                        "screen": "hard_candidate" if mean >= 0.7 else
                                  "soft_candidate" if mean >= 0.4 else "weak_signal"})
    return {"interpretation": "Descriptive stage-10 screen per signature and training mode, not proof of end-to-end learned hierarchy",
            "thresholds": {"hard_candidate": 0.7, "soft_candidate": 0.4}, "results": summary}


def main(argv=None):
    config = parse_args(argv)
    config.output.mkdir(parents=True, exist_ok=True)
    if any(config.output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {config.output}")
    torch.set_num_threads(config.torch_threads)
    torch.manual_seed(config.seeds[0])
    np.random.seed(config.seeds[0])
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    configuration = {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(config).items()}
    def git(*args):
        completed = subprocess.run(["git", *args], cwd=Path(__file__).resolve().parents[2],
                                   capture_output=True, text=True, check=False)
        return completed.stdout.strip() if completed.returncode == 0 else "unavailable"
    manifest = {"schema_version": 1, "status": "preparing_features", "experiment": "05_signature_validation",
                "run_classification": "smoke" if len(config.seeds) < 3 or max(config.stages) < 10 else
                    "subset_benchmark" if config.max_train_per_client or config.max_test_per_domain else "full_data_benchmark",
                "started_unix": time.time(), "config": configuration,
                "git": {"commit": git("rev-parse", "HEAD"), "working_tree_status": git("status", "--short")},
                "source": source_provenance(),
                "invocation": {"executable": sys.executable,
                    "argv": sys.argv if argv is None else [sys.argv[0], *argv], "cwd": str(Path.cwd())},
                "environment": {"python": platform.python_version(), "torch": torch.__version__,
                    "cuda": torch.version.cuda, "device": config.device,
                    "packages": package_versions(), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None},
                "protocol": {
                    "representation": "cached frozen eval-mode ResNet-18; deterministic resize and ImageNet normalization; no augmentation",
                    "training": "paired seed/shared head and partitions; Adam reset each round; homogeneous rank",
                    "signature_timing": "after local training and, in MH mode, after delta-W gossip",
                    "labels": "known domains construct benchmark shards and score clusters only; no labels in signatures or clustering",
                    "cluster_count": "prespecified; not inferred from adapters",
                    "topology": "flat topology client order is permuted by training seed, independently of domain labels",
                    "clustering_order": "seeded client permutation at each stage to remove domain-sorted tie-breaking",
                    "spectral_baseline": "original standardized singular values of A, B, BA plus factor norms; Ward linkage",
                    "row_norms": "L1 normalized per-output-row norms of scaled delta-W; cosine affinity; average linkage",
                    "inverse_delta_l2": "1/(1+L2 distance between flattened scaled delta-W); average linkage",
                    "evaluation": "both personalized client mean and merged consensus on the union of disjoint test shards",
                    "limitation": "offline diagnostic with global visibility of adapters; no distributed discovery overhead or end-to-end learned hierarchy is claimed"},
                "completed_runs": []}
    write_json(config.output / "manifest.json", manifest)
    records = []
    try:
        train, test, metadata = load_features(config.data_dir, config.feature_cache, torch.device(config.device),
                                              config.image_size, config.feature_batch_size, config.num_workers)
        manifest["feature_cache"], manifest["status"] = metadata, "running"
        write_json(config.output / "manifest.json", manifest)
        with (config.output / "summary.csv").open("w", newline="") as summary_file:
            writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            for seed in config.seeds:
                for method in config.methods:
                    record = run_one(config, seed, method, train, test, metadata, config.output)
                    records.append(record)
                    with (config.output / "results.jsonl").open("a") as handle:
                        handle.write(json.dumps(record, allow_nan=False) + "\n")
                    for stage in record["stages"]:
                        for signature in stage["signatures"]:
                            writer.writerow({"seed": seed, "method": method, "stage": stage["stage"],
                                **{key: signature[key] for key in ("signature", "adjusted_rand_index", "normalized_mutual_info")},
                                **{key: stage["evaluation"][key] for key in ("personalized_accuracy", "consensus_accuracy")}})
                    summary_file.flush()
                    manifest["completed_runs"].append({"seed": seed, "method": method, "file": f"seed{seed}_{method}.json"})
                    write_json(config.output / "manifest.json", manifest)
        write_json(config.output / "gate_g1.json", summarize_gate(records))
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        manifest["wall_seconds"] = time.time() - manifest["started_unix"]
        write_json(config.output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    main()
