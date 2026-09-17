#!/usr/bin/env python3
"""Source-pinned RoBERTa/GLUE quantity-skew decentralized LoRA experiment.

Independent Dec-LoRA reimplementation, not author code or an exact reported-score
replication. See docs/research/2026-09-17-quantity-skew-protocol.md for assumptions.
All peers run sequentially in one process with explicit graph-supported payloads.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import socket
import sys
import time

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.data.quantity_skew import partition_quantity_skew, partition_manifest
from src.federated.adaptive_rank import AdaptiveRankController, POLICY_PROVENANCE
from src.federated.compact_merge import merge_compact
from src.federated.domain_weights import weighted_metropolis
from src.federated.quantity_transport import tree_assemble
from src.models.roberta_lora import (inject_query_value, adapter_modules,
                                    export_state, install_state, expand_state, state_bytes)

MODEL_REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
DATA_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
ARMS = ("declora16", "declora4", "product16_sample", "fixed_uniform",
        "fixed_sample", "adaptive_uniform", "adaptive_sample")


def write_json(path, value):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def archive_source(output):
    repo = PROJECT.parent
    paths = [Path(__file__).resolve(), PROJECT / "requirements-transformer-gpu.txt"]
    paths += list((PROJECT / "src").rglob("*.py"))
    paths += [repo / "project-2-domain-aware-allocation/framework/aggregation/domain_weighting.py",
              repo / "project-2-domain-aware-allocation/experiment/experiment1/signals.py"]
    manifest = {}
    for path in sorted(set(paths)):
        relative = path.relative_to(repo)
        dest = output / "source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        manifest[str(relative)] = sha256(path)
    write_json(output / "source_manifest.json", manifest)
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def read_data(args, tokenizer):
    import pyarrow.parquet as parquet
    data, manifests = {}, {}
    for split in ("train", "validation"):
        path = args.assets / "glue" / args.task / f"{split}-00000-of-00001.parquet"
        table = parquet.read_table(path).to_pydict()
        if args.task == "sst2":
            encoded = tokenizer(table["sentence"], padding="max_length", truncation=True, max_length=args.max_length)
        else:
            encoded = tokenizer(table["sentence1"], table["sentence2"], padding="max_length", truncation=True, max_length=args.max_length)
        tensors = {key: torch.tensor(value, dtype=torch.long) for key, value in encoded.items()}
        tensors["labels"] = torch.tensor(table["label"], dtype=torch.long)
        source_ids = np.arange(len(table["label"]))
        if split == "train" and args.max_train:
            # Seeded smoke subset only; full-budget comparisons require max_train=0.
            source_ids = np.random.default_rng(args.seed + 901).permutation(len(source_ids))[:args.max_train]
            tensors = {key: value[source_ids] for key, value in tensors.items()}
        data[split] = tensors
        manifests[split] = {"file_sha256": sha256(path), "rows_in_file": len(table["label"]),
                            "rows_used": len(source_ids), "source_indices": source_ids.tolist(),
                            "source_indices_sha256": hashlib.sha256(source_ids.astype("<i8").tobytes()).hexdigest()}
    return data, manifests


def batch(data, indices, device):
    result = {key: value[indices] for key, value in data.items()}
    length = int(result["attention_mask"].sum(1).max())
    # Dynamic padding keeps full sequence content up to the declared max_length.
    return {key: (value[:, :length] if value.ndim == 2 else value).to(device)
            for key, value in result.items()}


@torch.no_grad()
def evaluate(model, data, batch_size, device):
    model.eval()
    predictions, total_loss = [], 0.0
    for start in range(0, len(data["labels"]), batch_size):
        inputs = batch(data, slice(start, start + batch_size), device)
        result = model(**inputs)
        total_loss += float(result.loss) * len(inputs["labels"])
        predictions.extend(result.logits.argmax(1).cpu().tolist())
    labels = data["labels"].numpy()
    pred = np.asarray(predictions)
    tp = int(np.sum((pred == 1) & (labels == 1)))
    fp = int(np.sum((pred == 1) & (labels != 1)))
    fn = int(np.sum((pred != 1) & (labels == 1)))
    correct = int(np.sum(pred == labels))
    return {"accuracy": 100.0 * correct / len(labels), "correct": correct, "n": len(labels),
            "f1": 200.0 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "loss": total_loss / len(labels), "predictions": predictions,
            "predictions_sha256": hashlib.sha256(pred.astype("<i8").tobytes()).hexdigest()}


def stable_probe(model, state, capacity, data, indices, seed, device, alpha):
    # Eval-mode TRAINING batch. Dropout is off; dedicated CPU generator for
    # added A rows means neither training RNG nor sample order is consumed.
    probe = expand_state(state, capacity, alpha, generator=torch.Generator().manual_seed(seed))
    install_state(model, probe)
    model.eval()
    model.zero_grad(set_to_none=True)
    result = model(**batch(data, indices, device))
    result.loss.backward()
    stable_ranks = []
    for module in adapter_modules(model).values():
        for parameter in (module.A, module.B):
            gradient = parameter.grad.detach().float()
            gram = gradient @ gradient.T if gradient.shape[0] <= gradient.shape[1] else gradient.T @ gradient
            largest = torch.linalg.eigvalsh(gram)[-1].clamp_min(1e-30)
            stable_ranks.append(float(gram.trace() / largest))
    record = {"stable_rank": float(np.median(stable_ranks)), "loss": float(result.loss),
              "quality": 1.0 / (1.0 + float(result.loss)), "probe_examples": len(indices),
              "probe_rank": capacity, "scope": "fixed training batch, eval mode, capability-rank probe"}
    model.zero_grad(set_to_none=True)
    return record


def resize_state(state, rank, alpha, seed):
    current = next(iter(state["adapter"].values()))["A"].shape[0]
    if rank >= current:
        return expand_state(state, rank, alpha, generator=torch.Generator().manual_seed(seed))
    adapter, _ = merge_compact([state["adapter"]], [1.0], rank, alpha)
    return {"adapter": adapter, "head": {k: v.clone() for k, v in state["head"].items()}}


def train_local(model, state, data, indices, args, cid, round_index, device):
    install_state(model, state)
    model.train()
    seed_all(args.seed + 100003 * (round_index + 1) + 997 * cid)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    rng = np.random.default_rng(args.seed + 200003 * (round_index + 1) + 991 * cid)
    steps, seen, loss_sum, grad_norm_max = 0, 0, 0.0, 0.0
    order_digest = hashlib.sha256()
    def batches():
        if args.local_steps:
            # Stream reshuffled local data across epoch boundaries. Every peer
            # takes exactly K steps of exactly B examples, regardless of n_i.
            # No cross-client examples are ever introduced.
            remaining = np.empty(0, dtype=np.int64)
            for _ in range(args.local_steps):
                while len(remaining) < args.batch_size:
                    remaining = np.concatenate((remaining, rng.permutation(indices)))
                yield remaining[:args.batch_size]
                remaining = remaining[args.batch_size:]
        else:
            for _ in range(args.local_epochs):
                order = rng.permutation(indices)
                for start in range(0, len(order), args.batch_size):
                    yield order[start:start + args.batch_size]
    for selected in batches():
        inputs = batch(data, selected, device)
        optimizer.zero_grad(set_to_none=True)
        result = model(**inputs)
        if not torch.isfinite(result.loss):
            raise FloatingPointError(f"nonfinite loss peer={cid} round={round_index}")
        result.loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        steps += 1
        seen += len(selected)
        loss_sum += float(result.loss) * len(selected)
        grad_norm_max = max(grad_norm_max, float(norm))
        order_digest.update(selected.astype("<i8").tobytes())
    final = export_state(model)
    optimizer.zero_grad(set_to_none=True)
    return final, {"steps": steps, "examples": seen, "mean_loss": loss_sum / seen,
                   "max_preclip_gradient_norm": grad_norm_max, "sample_order_sha256": order_digest.hexdigest()}


def effective_dot(left, right, alpha):
    value = 0.0
    for name, pair in left.items():
        a, b = pair["A"].double(), pair["B"].double()
        c, d = right[name]["A"].double(), right[name]["B"].double()
        value += float(torch.sum((b.T @ d) * (a @ c.T))) * alpha ** 2 / (len(a) * len(c))
    return value


def weighted_disagreement(states, weights, alpha):
    gram = np.empty((len(states), len(states)))
    for i in range(len(states)):
        for j in range(i + 1):
            gram[i, j] = gram[j, i] = effective_dot(states[i]["adapter"], states[j]["adapter"], alpha)
    weights = np.asarray(weights)
    mean_norm_sq = float(weights @ gram @ weights)
    variance = max(0.0, float(weights @ np.diag(gram) - mean_norm_sq))
    return {"adapter_variance": variance, "mean_adapter_norm_sq": mean_norm_sq,
            "relative_adapter_variance": variance / max(mean_norm_sq, 1e-30)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--task", choices=("sst2", "mrpc"), default="sst2")
    parser.add_argument("--partition", choices=("equal", "quantity"), default="quantity")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--local-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if any(v <= 0 for v in (args.rounds, args.local_epochs, args.batch_size, args.max_length, args.threads)) or args.local_steps < 0 or args.max_train < 0:
        parser.error("invalid nonpositive budget or negative subset/step count")
    if socket.gethostname().split(".")[0] != "gpu003":
        raise RuntimeError("Scientific execution is restricted to authorized gpu003")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    seed_all(args.seed)
    from transformers import AutoTokenizer, RobertaForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(args.assets / "roberta-base", local_files_only=True)
    model = RobertaForSequenceClassification.from_pretrained(args.assets / "roberta-base", num_labels=2,
                                                            local_files_only=True, attn_implementation="eager")
    model = inject_query_value(model, rank=16, alpha=args.alpha, dropout=0.0).to(device)
    data, data_manifest = read_data(args, tokenizer)
    output = args.output
    if args.verify_only:
        checkpoint = torch.load(output / "final.pt", map_location="cpu", weights_only=False)
        install_state(model, checkpoint["state"])
        metrics = evaluate(model, data["validation"], args.batch_size, device)
        previous = json.loads((output / "summary.json").read_text())
        if metrics["predictions_sha256"] != previous["final"]["predictions_sha256"]:
            raise AssertionError("independent process final checkpoint predictions do not match")
        write_json(output / "fresh_process_verification.json", {"host": socket.gethostname(), "checkpoint_sha256": sha256(output / "final.pt"), "metrics": metrics, "matches": True})
        print(json.dumps({"verified": True, "accuracy": metrics["accuracy"], "output": str(output)}), flush=True)
        return
    output.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_json(output / "config.json", config)
    source_sha = archive_source(output)
    shares = [1] * 10 if args.partition == "equal" else [1, 1, 1, 1, 2, 2, 2, 4, 4, 4]
    labels = data["train"]["labels"].numpy()
    shards = partition_quantity_skew(labels, client_fractions=shares, seed=args.seed + 10000, assignment_seed=args.seed + 20000)
    split = partition_manifest(labels, shards, seed=args.seed + 10000, assignment_seed=args.seed + 20000,
                               dataset_id=f"nyu-mll/glue@{DATA_REVISION}/{args.task}/train")
    split["data_files"] = data_manifest
    split["validation_labels"] = data["validation"]["labels"].tolist()
    write_json(output / "split.json", split)
    counts = np.asarray([len(shards[i]) for i in range(10)])
    sample_weights = counts / counts.sum()
    capacity = np.random.default_rng(args.seed + 30000).permutation([4, 4, 4, 4, 8, 8, 8, 16, 16, 16]).tolist()
    if args.arm == "declora16" or args.arm == "product16_sample":
        ranks = [16] * 10
    elif args.arm == "declora4":
        ranks = [4] * 10
    else:
        ranks = list(capacity)
    weights = sample_weights if args.arm.endswith("sample") else np.ones(10) / 10
    proposal = np.zeros((10, 10))
    neighbors = {i: sorted({(i - 1) % 10, (i + 1) % 10}) for i in range(10)}
    for i in range(10):
        proposal[i, [i, *neighbors[i]]] = 1.0 / 3
    matrix = weighted_metropolis(proposal, weights)
    similarity = np.sqrt(weights[:, None]) * matrix / np.sqrt(weights[None, :])
    eigenvalues = np.linalg.eigvalsh(similarity)
    if not np.allclose(weights @ matrix, weights, atol=1e-12):
        raise AssertionError("sample-weight stationarity failed")
    # Deployment needs room for rank16. Choose the first capability16 peer
    # before training, independently of data quantities or validation scores.
    deployment_root = capacity.index(16)
    count_messages = [{"sender": i, "receiver": j, "training_count": int(counts[i]), "bytes": 16}
                      for i in range(10) for j in neighbors[i]]
    graph = {"neighbors": neighbors, "matrix": matrix.tolist(), "stationary_weights": weights.tolist(),
             "sample_weights": sample_weights.tolist(), "capacity": capacity,
             "eigenvalues": eigenvalues.tolist(), "absolute_spectral_gap": float(1 - max(abs(eigenvalues[:-1]))),
             "root": deployment_root, "setup_count_messages": count_messages, "setup_bytes": sum(v["bytes"] for v in count_messages),
             "quantity_assignment_seed": args.seed + 20000, "capacity_assignment_seed": args.seed + 30000}
    write_json(output / "graph.json", graph)
    base = export_state(model)
    # All ranks share the same zero effective initialization and common A rows.
    states = {}
    for cid, rank in enumerate(ranks):
        states[cid] = {"adapter": {name: {"A": pair["A"][:rank].clone(), "B": pair["B"][:, :rank].clone()}
                                    for name, pair in base["adapter"].items()},
                       "head": {key: value.clone() for key, value in base["head"].items()}}
    controllers = {i: AdaptiveRankController({4:16, 8:64, 16:256}[capacity[i]], initial_rank="max", warmup_rounds=2)
                   for i in range(10)} if args.arm.startswith("adaptive") else {}
    probe_indices = {i: np.random.default_rng(args.seed + 40000 + i).permutation(shards[i])[:args.batch_size]
                     for i in range(10)}
    environment = {"host": socket.gethostname(), "python": platform.python_version(), "torch": torch.__version__,
                   "gpu": torch.cuda.get_device_name(), "model_revision": MODEL_REVISION, "dataset_revision": DATA_REVISION,
                   "model_files": {p.name: sha256(p) for p in (args.assets / "roberta-base").iterdir() if p.is_file()},
                   "source_sha256": source_sha, "simulation": "single-process sequential peers; graph-restricted payloads",
                   "adapter_policy": POLICY_PROVENANCE, "base_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters() if not p.requires_grad),
                   "optimizer_policy": "AdamW weight_decay=0, reset for every client/round in all arms; lr constant; grad clip1",
                   "precision": "fp32; deterministic algorithms", "evaluation_split": "official labeled validation; not a hidden test"}
    write_json(output / "environment.json", environment)
    from src.federated.quantity_transport import neighbor_round
    mode = "factor" if args.arm.startswith("declora") else "effective"
    metric = "accuracy" if args.task == "sst2" else "f1"
    history, best, cumulative_training_bytes, cumulative_evaluation_bytes = [], None, 0, 0
    started = time.monotonic()
    for round_index in range(args.rounds):
        round_started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        local_records, probes = {}, {}
        for cid in range(10):
            if cid in controllers:
                probe = stable_probe(model, states[cid], capacity[cid], data["train"], probe_indices[cid], args.seed + 50000 + cid,
                                     device, args.alpha)
                ranks[cid] = controllers[cid].update(probe["stable_rank"])
                states[cid] = resize_state(states[cid], ranks[cid], args.alpha, args.seed + 60000 + cid + 10 * round_index)
                probes[cid] = {"before": probe, "controller": controllers[cid].diagnostics()}
            states[cid], local_records[cid] = train_local(model, states[cid], data["train"], np.asarray(shards[cid]), args, cid, round_index, device)
            if cid in controllers:
                model.eval()
                with torch.no_grad():
                    quality_loss = float(model(**batch(data["train"], probe_indices[cid], device)).loss)
                controllers[cid].observe_quality(1 / (1 + quality_loss))
                probes[cid]["after_training_loss"] = quality_loss
        states, gossip = neighbor_round(states, matrix, neighbors, dict(enumerate(ranks)), args.alpha, mode=mode)
        assembled, assembly = tree_assemble(states, {i: float(weights[i]) for i in range(10)}, neighbors, root_id=deployment_root,
                                            mode=mode, target_rank=(ranks[0] if mode == "factor" else 16), alpha=args.alpha,
                                            disseminate=False)
        install_state(model, assembled)
        evaluation = evaluate(model, data["validation"], args.batch_size, device)
        cumulative_training_bytes += gossip["bytes"]
        cumulative_evaluation_bytes += assembly["bytes"]
        record = {"round": round_index + 1, "ranks": list(ranks), "local": local_records, "probes": probes,
                  "validation": evaluation, "gossip": gossip, "evaluation_assembly": assembly,
                  "cumulative_training_bytes": cumulative_training_bytes, "cumulative_evaluation_assembly_bytes": cumulative_evaluation_bytes,
                  "state_bytes": {i: state_bytes(states[i]) for i in range(10)},
                  "weighted_disagreement": weighted_disagreement(states, weights, args.alpha),
                  "peak_process_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_process_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "round_seconds": time.monotonic() - round_started, "elapsed_seconds": time.monotonic() - started}
        history.append(record)
        with open(output / "rounds.jsonl", "a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        checkpoint = {"state": assembled, "round": round_index + 1, "source_sha256": source_sha, "config": config}
        torch.save(checkpoint, output / "final.pt")
        if best is None or evaluation[metric] > best["validation"][metric]:
            best = {"round": round_index + 1, "validation": evaluation}
            torch.save(checkpoint, output / "best.pt")
        print(json.dumps({"arm": args.arm, "seed": args.seed, "round": round_index + 1, metric: evaluation[metric],
                          "ranks": ranks, "seconds": round(record["round_seconds"], 2)}), flush=True)
    # The final evaluated assembly is already at the designated capable peer.
    # No rank16 model is installed at capacity4/8 peers. Earlier assemblies are
    # evaluation-only and never feed back into local training.
    delivery = history[-1]["evaluation_assembly"]
    summary = {"status": "complete", "arm": args.arm, "seed": args.seed, "partition": args.partition,
               "primary_metric": metric, "best": best, "final": history[-1]["validation"], "rounds": args.rounds,
               "training_bytes": cumulative_training_bytes, "evaluation_assembly_bytes": cumulative_evaluation_bytes,
               "setup_bytes": graph["setup_bytes"],
               "production_bytes_setup_training_final_assembly": graph["setup_bytes"] + cumulative_training_bytes + delivery["bytes"],
               "all_experiment_bytes_including_evaluations": graph["setup_bytes"] + cumulative_training_bytes + cumulative_evaluation_bytes,
               "final_delivery": delivery, "total_steps": sum(v["steps"] for r in history for v in r["local"].values()),
               "total_examples": sum(v["examples"] for r in history for v in r["local"].values()),
               "total_probe_examples": sum(v["before"]["probe_examples"] * 2 for r in history for v in r["probes"].values()),
               "elapsed_seconds": time.monotonic() - started, "final_checkpoint_sha256": sha256(output / "final.pt"),
               "best_checkpoint_sha256": sha256(output / "best.pt"), "source_sha256": source_sha,
               "interpretation": "independent paper-based implementation; changed quantity partition; no claim against printed scores"}
    write_json(output / "summary.json", summary)


if __name__ == "__main__":
    main()
