"""Positive control: exact peer gradient synchronization versus pooled LoRA.

This is a one-process simulation, with uniform rank 16 at every peer. Each
pooled minibatch is split by the original non-IID ownership; only local mean
factor gradients and sample counts cross explicitly recorded neighbor edges.
After allreduce, every peer has the same gradient and Adam state. Identical
parameter/optimizer replicas are represented by one object, rather than
executing the same optimizer operation fifteen times. This control does not
solve heterogeneous-rank training or establish privacy. Its deliberately high
communication cost is recorded, including a reduction and broadcast every
minibatch. Data/example indices never appear in communication records.
"""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.feature_cache import load_features, tensor_digest, write_json
from experiments.protocol_benchmark import (
    FeatureClient, initial_parameters, make_splits, package_versions, topology_order,
)
from src.federated.mixing import build_topology


def spanning_tree(neighbors, root):
    parent, order = {root: None}, [root]
    for sender in order:
        for receiver in sorted(neighbors[sender]):
            if receiver not in parent:
                parent[receiver] = sender
                order.append(receiver)
    if len(order) != len(neighbors):
        raise ValueError("peer graph is disconnected")
    edges = [(child, parent[child]) for child in reversed(order[1:])]
    if any(receiver not in neighbors[sender] for sender, receiver in edges):
        raise AssertionError("tree message leaves the neighbor graph")
    return parent, order


def tree_gradient_allreduce(local, counts, neighbors, root):
    """Sample-weighted mean of local mean gradients, by peer messages only."""
    parent, order = spanning_tree(neighbors, root)
    if set(local) != set(order) or set(counts) != set(order):
        raise ValueError("every peer must provide a gradient and sample count")
    template = local[root]
    if any(g.shape != template.shape or g.dtype != template.dtype for g in local.values()):
        raise ValueError("gradient payload shape/dtype mismatch")
    accumulators = {cid: local[cid].clone().mul_(counts[cid]) for cid in order}
    masses = {cid: int(counts[cid]) for cid in order}
    if any(n < 0 for n in masses.values()) or not sum(masses.values()):
        raise ValueError("sample counts must be nonnegative with positive total")
    # A parent sees only an incoming subtree gradient sum and its sample count.
    for child in reversed(order[1:]):
        receiver = parent[child]
        payload = accumulators.pop(child)
        accumulators[receiver].add_(payload)
        masses[receiver] += masses[child]
    averaged = accumulators[root].div_(masses[root])
    # Each message follows one tree edge; every peer receives the root mean.
    received = {root: averaged}
    received_counts = {root: masses[root]}
    for receiver in order[1:]:
        received[receiver] = received[parent[receiver]].clone()
        received_counts[receiver] = received_counts[parent[receiver]]
    if not all(torch.equal(averaged, value) for value in received.values()):
        raise AssertionError("allreduce recipients disagree")
    if any(value != masses[root] for value in received_counts.values()):
        raise AssertionError("allreduce sample-count recipients disagree")
    return averaged, masses[root]


def factor_parameters(model):
    return (model.lora_A, model.lora_B)


def flatten_gradient(model):
    return torch.cat([parameter.grad.detach().reshape(-1) for parameter in factor_parameters(model)])


def install_gradient(model, gradient):
    offset = 0
    for parameter in factor_parameters(model):
        size = parameter.numel()
        parameter.grad = gradient[offset:offset + size].view_as(parameter).clone()
        offset += size
    if offset != gradient.numel():
        raise ValueError("gradient payload length mismatch")


@torch.no_grad()
def evaluate(model, test, batch_size):
    model.eval()
    predictions = []
    for start in range(0, len(test["labels"]), batch_size):
        predictions.append(model(test["features"][start:start + batch_size]).argmax(1))
    predictions = torch.cat(predictions)
    return predictions, int((predictions == test["labels"]).sum())


def run_seed(args, seed, train, test, metadata):
    args.current_seed = seed
    start = time.perf_counter()
    splits, assignments = make_splits(train["labels"].cpu().numpy(), test["labels"].cpu().numpy(), args, seed)
    ids = sorted(splits)
    neighbors = build_topology(topology_order(assignments, seed), "ring")
    root = ids[0]
    parent, order = spanning_tree(neighbors, root)
    owners = torch.full((len(train["labels"]),), -1, dtype=torch.long, device=args.device)
    for cid in ids:
        indices = torch.tensor(splits[cid]["train_indices"], device=args.device)
        if torch.any(owners[indices] != -1):
            raise AssertionError("training ownership overlaps")
        owners[indices] = cid
    if torch.any(owners < 0):
        raise AssertionError("training ownership omits data")
    initial = initial_parameters(train["features"].shape[1], args.reference_rank, seed)
    whole = {"train_indices": list(range(len(train["labels"]))),
             "test_indices": list(range(len(test["labels"])))}
    pooled_client = FeatureClient(0, 0, initial, args.reference_rank, args, train, test, whole, torch.device(args.device))
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    pooled = pooled_client.model.to(dtype=dtype)
    distributed = copy.deepcopy(pooled)
    initial_hash = tensor_digest(*[v for v in pooled.state_dict().values()])
    make_optimizer = lambda model: torch.optim.Adam(factor_parameters(model), lr=args.lr, weight_decay=args.weight_decay)
    pooled_optimizer, distributed_optimizer = make_optimizer(pooled), make_optimizer(distributed)
    factor_count = sum(p.numel() for p in factor_parameters(distributed))
    gradient_bytes = factor_count * next(iter(factor_parameters(distributed))).element_size()
    # Both phases carry a gradient payload and one int64 sample count.
    messages_per_step = 2 * (len(ids) - 1)
    bytes_per_step = messages_per_step * (gradient_bytes + 8)
    path = args.output / f"seed{seed}.json"
    record = {
        "schema_version": 1, "status": "running", "seed": seed,
        "execution": "single-process simulation with collapsed identical model/Adam replicas",
        "method": "uniform-rank exact peer factor-gradient allreduce",
        "scope": "positive control, not heterogeneous/adaptive solution or privacy proof",
        "dtype": args.dtype, "alpha": args.alpha, "rank": args.reference_rank,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "feature_cache_identity_sha256": metadata["cache_identity_sha256"],
        "initial_state_sha256": initial_hash,
        "split_sha256": hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest(),
        "ownership_sha256": tensor_digest(owners),
        "n_train": len(train["labels"]), "n_test": len(test["labels"]),
        "optimizer": {"name": "Adam", "lr": args.lr, "weight_decay": args.weight_decay,
                      "reset_each_epoch": args.reset_each_epoch, "batch_size": args.batch_size},
        "minibatch_schedule": "FeatureClient cid0 seeded generator; pooled batch split by original client ownership",
        "topology": neighbors, "root": root,
        "reduction_edges": [[child, parent[child]] for child in reversed(order[1:])],
        "broadcast_edges": [[parent[child], child] for child in order[1:]],
        "message_payload": "factor gradient sum/mean and int64 sample count; no raw examples or example indices",
        "gradient_bytes_per_message": gradient_bytes,
        "messages_per_step": messages_per_step, "bytes_per_step": bytes_per_step,
        "communication_exclusions": "initial model dissemination, schedule metadata, transport framing; oracle/evaluation are diagnostics",
        "factor_coordinate_policy": "shared initialization, no projection/SVD, persistent identical factor coordinates",
        "proof_tolerance": {"dtype": "float64", "atol": 2e-11, "rtol": 2e-10, "checked_batches": args.proof_batches},
        "proof": [], "rounds": [],
    }
    write_json(path, record)
    total_steps = 0
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        if epoch and args.reset_each_epoch:
            pooled_optimizer, distributed_optimizer = make_optimizer(pooled), make_optimizer(distributed)
        pooled.train(); distributed.train()
        permutation = torch.randperm(len(train["labels"]), generator=pooled_client.rng).to(args.device)
        losses = [0., 0.]
        epoch_steps = 0
        for begin in range(0, len(permutation), args.batch_size):
            batch = permutation[begin:begin + args.batch_size]
            labels = train["labels"][batch]
            pooled_optimizer.zero_grad(set_to_none=True)
            pooled_loss = nn.functional.cross_entropy(pooled(train["features"][batch]), labels)
            pooled_loss.backward()
            local, counts = {}, {}
            weighted_loss = 0.
            for cid in ids:
                local_indices = batch[owners[batch] == cid]
                count = len(local_indices)
                counts[cid] = count
                if count:
                    distributed.zero_grad(set_to_none=True)
                    local_loss = nn.functional.cross_entropy(distributed(train["features"][local_indices]), train["labels"][local_indices])
                    local_loss.backward()
                    local[cid] = flatten_gradient(distributed)
                    weighted_loss += float(local_loss.detach()) * count
                else:
                    local[cid] = torch.zeros(factor_count, dtype=dtype, device=args.device)
            aggregate, count = tree_gradient_allreduce(local, counts, neighbors, root)
            if count != len(batch):
                raise AssertionError("peer message counts do not reconstruct pooled batch")
            if total_steps < args.proof_batches:
                # Independent fp64 same-state oracle isolates aggregation error
                # from float32 accumulation and the two optimizer trajectories.
                proof_model = copy.deepcopy(distributed).double()
                proof_local = {}
                for cid in ids:
                    proof_indices = batch[owners[batch] == cid]
                    if len(proof_indices):
                        proof_model.zero_grad(set_to_none=True)
                        proof_loss = nn.functional.cross_entropy(proof_model(train["features"][proof_indices].double()), train["labels"][proof_indices])
                        proof_loss.backward()
                        proof_local[cid] = flatten_gradient(proof_model)
                    else:
                        proof_local[cid] = torch.zeros(factor_count, dtype=torch.float64, device=args.device)
                proof_aggregate, proof_count = tree_gradient_allreduce(proof_local, counts, neighbors, root)
                proof_model.zero_grad(set_to_none=True)
                oracle_loss = nn.functional.cross_entropy(proof_model(train["features"][batch].double()), labels)
                oracle_loss.backward()
                oracle = flatten_gradient(proof_model)
                error = float((oracle - proof_aggregate).abs().max())
                torch.testing.assert_close(proof_aggregate, oracle, atol=2e-11, rtol=2e-10)
                if proof_count != count:
                    raise AssertionError("proof batch counts differ")
                record["proof"].append({"step": total_steps + 1, "max_gradient_abs_error": error,
                                        "n_samples": count, "contributing_peers": sum(v > 0 for v in counts.values())})
                del proof_model, proof_local, proof_aggregate, oracle
            install_gradient(distributed, aggregate)
            if not torch.isfinite(aggregate).all() or not torch.isfinite(pooled_loss):
                raise AssertionError("nonfinite optimization state")
            pooled_optimizer.step(); distributed_optimizer.step()
            losses[0] += float(pooled_loss.detach()) * count
            losses[1] += weighted_loss
            total_steps += 1
            epoch_steps += 1
        pooled_prediction, pooled_correct = evaluate(pooled, test, args.eval_batch_size)
        distributed_prediction, distributed_correct = evaluate(distributed, test, args.eval_batch_size)
        max_factor_error = max(float((a - b).abs().max()) for a, b in zip(factor_parameters(pooled), factor_parameters(distributed)))
        row = {
            "epoch": epoch + 1, "optimizer_steps": epoch_steps,
            "pooled_train_loss": losses[0] / len(train["labels"]),
            "distributed_train_loss": losses[1] / len(train["labels"]),
            "pooled_full_test_correct": pooled_correct,
            "distributed_full_test_correct": distributed_correct,
            "pooled_full_test_accuracy": pooled_correct / len(test["labels"]),
            "distributed_full_test_accuracy": distributed_correct / len(test["labels"]),
            "test_prediction_disagreements": int((pooled_prediction != distributed_prediction).sum()),
            "max_factor_abs_error": max_factor_error,
            "train_sample_exposures_per_arm": len(train["labels"]),
            "peer_messages": epoch_steps * messages_per_step,
            "peer_payload_bytes": epoch_steps * bytes_per_step,
            "wall_seconds": time.perf_counter() - epoch_start,
        }
        record["rounds"].append(row)
        write_json(path, record)
        print(f"seed={seed} epoch={epoch+1}/{args.epochs} pooled={row['pooled_full_test_accuracy']:.4f} "
              f"peer={row['distributed_full_test_accuracy']:.4f} disagreements={row['test_prediction_disagreements']} "
              f"factor_error={max_factor_error:.3e} seconds={row['wall_seconds']:.2f}", flush=True)
    record.update({"status": "completed", "total_optimizer_steps_per_arm": total_steps,
                   "total_peer_messages": total_steps * messages_per_step,
                   "total_peer_payload_bytes": total_steps * bytes_per_step,
                   "total_wall_seconds": time.perf_counter() - start,
                   "max_proof_gradient_abs_error": max(r["max_gradient_abs_error"] for r in record["proof"])})
    checkpoint = args.output / f"seed{seed}_checkpoint.pt"
    torch.save({"pooled": pooled.state_dict(), "distributed": distributed.state_dict(),
                "alpha": args.alpha, "rank": args.reference_rank, "seed": seed,
                "feature_cache_identity_sha256": metadata["cache_identity_sha256"]}, checkpoint)
    record["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    write_json(path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--proof-batches", type=int, default=64)
    parser.add_argument("--reset-each-epoch", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = parser.parse_args()
    if args.epochs < 1 or args.proof_batches < 1:
        parser.error("epochs and proof batches must be positive")
    args.alpha, args.reference_rank = 32., 16
    args.lr, args.weight_decay = .001, .0001
    args.batch_size, args.eval_batch_size, args.local_epochs = 128, 2048, 1
    args.n_domains, args.clients_per_domain, args.dirichlet_alpha = 5, 3, .5
    args.max_train_per_client, args.max_test_per_domain = 0, 0
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "packages": package_versions(), "host": __import__("socket").gethostname()}
    write_json(args.output / "manifest.json", manifest)
    try:
        train, test, metadata = load_features(args.data_dir, args.feature_cache, args.device)
        for dataset in (train, test):
            dataset["features"] = dataset["features"].to(device=args.device, dtype=torch.float32 if args.dtype == "float32" else torch.float64)
            dataset["labels"] = dataset["labels"].to(args.device)
        results = [run_seed(args, seed, train, test, metadata) for seed in args.seeds]
        manifest["status"] = "completed"
        manifest["results"] = [{"seed": r["seed"], "final": r["rounds"][-1],
                                "max_proof_gradient_abs_error": r["max_proof_gradient_abs_error"]} for r in results]
        write_json(args.output / "manifest.json", manifest)
    except BaseException as error:
        manifest.update({"status": "failed", "failure_type": type(error).__name__, "failure": str(error)})
        write_json(args.output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
