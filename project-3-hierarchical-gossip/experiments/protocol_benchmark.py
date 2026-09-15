"""Paired-seed CIFAR-100 benchmark using the validated delta-W runner.

Only cached features cross this module's training boundary. The shared frozen
backbone and random frozen classification head are identical across every client
and method in a seed. No test labels are used in training or model selection.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

from experiments.feature_cache import load_features, tensor_digest, write_json
from src.federated.hierarchical import two_tier_mixing, two_tier_message_cost, window_product, sinkhorn
from src.federated.merge import lora_to_delta, factorize_delta
from src.federated.mixing import build_topology, metropolis_hastings, spectral_gap
from src.federated.runner import DecentralizedRunner
from src.clustering.discovery import AdaptiveAffinityMixer, OnlineDomainDiscovery


METHODS = ("local", "fedavg", "mh", "oracle", "adaptive", "weighted", "adaptive_weighted",
           "adaptive_rank", "adaptive_weighted_rank")
SUMMARY_FIELDS = ["seed", "method", "merge", "rank_schedule", "error_feedback", "rounds",
                  "personalized_accuracy", "personalized_sample_weighted_accuracy",
                  "consensus_accuracy", "accuracy_gap", "worst_domain_accuracy",
                  "consensus_distance", "total_effective_messages", "total_effective_floats",
                  "total_operational_messages", "total_operational_floats",
                  "wall_seconds", "initial_state_sha256", "split_sha256"]


def conservative_factors(values, blend=0.10, bound=0.15):
    """Bounded, mean-one factors from a per-client signal."""
    x = np.asarray(values, dtype=float)
    if x.size == 0 or not np.isfinite(x).all() or x.std() < 1e-12:
        return np.ones(len(x), dtype=float)
    z = np.clip((x - x.mean()) / x.std(), -3.0, 3.0)
    raw = np.exp(np.clip(z, -4.0, 4.0)); raw /= raw.mean()
    f = 1.0 + blend * (raw - 1.0)
    # Scale after clipping so the factors preserve the base update scale.
    lo, hi = 0.0, 2.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if np.clip(f * mid, 1-bound, 1+bound).mean() < 1.0: lo = mid
        else: hi = mid
    return np.clip(f * ((lo + hi) / 2), 1-bound, 1+bound)


class DomainWeightedMixer:
    """Stateful symmetric DS mixer using update-norm domain evidence."""
    def __init__(self, base_mixer, n_clients, blend=0.10, bound=0.15, signal_values=None):
        self.base_mixer, self.n_clients = base_mixer, int(n_clients)
        self.blend, self.bound, self._factors = float(blend), float(bound), np.ones(n_clients)
        self.signal_values = None if signal_values is None else np.asarray(signal_values, dtype=float)

    def update(self, states, alpha, client_ids=None):
        if self.signal_values is None:
            self.signal_values = np.asarray(
                [sum(float(torch.linalg.vector_norm(d)) for d in lora_to_delta(s, alpha).values()) for s in states], dtype=float)
        self._factors = conservative_factors(self.signal_values, self.blend, self.bound)
        observe = getattr(self.base_mixer, "update", None)
        if observe is not None: observe(states, alpha=alpha, client_ids=client_ids)

    def __call__(self, round_idx):
        base = np.asarray(self.base_mixer(round_idx), dtype=float)
        scale = np.sqrt(np.outer(self._factors, self._factors))
        return sinkhorn(base * scale)


def source_provenance():
    """Identify the actual snapshot even when tar/scp deployment omits .git."""
    root = Path(__file__).resolve().parents[1]
    hashes = {}
    for directory in (root / "experiments", root / "src"):
        for path in sorted(directory.rglob("*.py")):
            hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    combined = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"files_sha256": hashes, "combined_sha256": combined}


def package_versions():
    versions = {}
    for package in ("torch", "torchvision", "numpy", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def topology_order(assignments, seed):
    """Remove the domain ordering encoded by sequential client identifiers."""
    return np.random.default_rng(seed + 7919).permutation(sorted(assignments)).tolist()


def build_mixing(method, assignments, bridge_every, topology="ring", seed=42, alpha=32.0):
    ids = sorted(assignments)
    n = len(ids)
    if method == "local":
        matrix = np.eye(n)
    elif method == "fedavg":
        # Client-uniform FedAvg is the centralized counterpart of the uniform
        # client objective preserved by doubly stochastic gossip.
        matrix = np.full((n, n), 1.0 / n)
    elif method in {"mh", "adaptive_rank"}:
        matrix = metropolis_hastings(build_topology(topology_order(assignments, seed), topology), client_ids=ids)
    elif method == "oracle":
        return lambda round_idx: two_tier_mixing(assignments, round_idx,
            bridge_every=bridge_every, client_ids=ids)
    elif method in {"adaptive", "adaptive_weighted"}:
        # Label-free stateful mixer. It observes only effective updates and
        # infers the number of groups by silhouette score before emitting a
        # soft Sinkhorn doubly-stochastic matrix on the configured topology.
        mixer = AdaptiveAffinityMixer(ids, alpha=float(alpha),
            discovery=OnlineDomainDiscovery(beta=0.8, min_clusters=2, max_clusters=8,
                                            temperature=0.5, self_weight=0.2),
            topology=topology,
            topology_client_ids=topology_order(assignments, seed))
        return DomainWeightedMixer(mixer, n, blend=0.10, bound=0.15) if method == "adaptive_weighted" else mixer
    elif method == "weighted":
        matrix = metropolis_hastings(
            build_topology(topology_order(assignments, seed), topology), client_ids=ids)
        mixer = lambda round_idx: matrix
        return DomainWeightedMixer(mixer, n, blend=0.10, bound=0.15)
    elif method == "adaptive_weighted_rank":
        matrix = metropolis_hastings(build_topology(topology_order(assignments, seed), topology), client_ids=ids)
        mixer = lambda round_idx: matrix
        return DomainWeightedMixer(mixer, n, blend=0.10, bound=0.15)
    else:
        raise ValueError(f"unknown method {method!r}")
    return lambda round_idx: matrix


def zero_pad_factor_mix(states, matrix, target_ranks):
    """Intentionally naive factor baseline: pad, average A/B, trim to receiver.

    No gauge alignment or alpha/r correction is introduced. The resulting
    effective update generally differs from the weighted delta-W target; that
    discrepancy is measured by FactorRunner, not called SVD truncation loss.
    """
    rank = max([p["A"].shape[0] for state in states for p in state.values()]
               + list(target_ranks))
    result = []
    for i, target in enumerate(target_ranks):
        state = {}
        for layer in states[0]:
            ref = states[0][layer]
            a = ref["A"].new_zeros((rank, ref["A"].shape[1]))
            b = ref["B"].new_zeros((ref["B"].shape[0], rank))
            for j, source in enumerate(states):
                r = source[layer]["A"].shape[0]
                a[:r] += float(matrix[i, j]) * source[layer]["A"]
                b[:, :r] += float(matrix[i, j]) * source[layer]["B"]
            state[layer] = {"A": a[:target].clone(), "B": b[:, :target].clone()}
        result.append(state)
    return result


class FactorRunner(DecentralizedRunner):
    def gossip_round(self, round_idx, states):
        if self.error_feedback:
            raise ValueError("error feedback is defined only for delta merging")
        matrix = self._mixing_matrix(round_idx)
        new_states = zero_pad_factor_mix(states, matrix,
                                        [self.target_ranks[cid] for cid in self.client_ids])
        before = [lora_to_delta(state, self.alpha) for state in states]
        after = [lora_to_delta(state, self.alpha) for state in new_states]
        energies, relative = [], []
        for i in range(len(states)):
            energy, ratios = 0.0, []
            for layer in before[0]:
                target = sum(float(matrix[i, j]) * before[j][layer] for j in range(len(states)))
                error = float(torch.sum((target - after[i][layer]) ** 2))
                norm = float(torch.sum(target ** 2))
                energy += error
                # Relative error is undefined if an exact cancellation makes
                # the target zero but factor mixing creates a nonzero update.
                ratios.append(error / norm if norm else (0.0 if error == 0.0 else None))
            energies.append(energy)
            relative.append(float(np.mean(ratios)) if all(x is not None for x in ratios) else None)
        messages, floats = self._communication(matrix, states)
        return new_states, {"messages": messages, "floats": floats,
                            "mean_tail_mass": None, "max_tail_mass": None,
                            "mean_residual_energy": None,
                            "mean_merge_error_energy": float(np.mean(energies)),
                            "mean_relative_merge_error": float(np.mean(relative)) if all(x is not None for x in relative) else None,
                            "consensus_distance": self._consensus_distance(new_states)}


class FeatureClient:
    """Head-only client with independent deterministic minibatch randomness."""
    def __init__(self, cid, domain, initial, rank, config, train, test, split, device):
        from src.models.lora_resnet import LoRALinear
        self.client_id, self.domain_id = cid, domain
        self.rank, self.alpha = rank, config.alpha
        self.device, self.config = device, config
        self.train_data, self.test_data = train, test
        self.train_indices = torch.tensor(split["train_indices"], device=device)
        self.test_indices = torch.tensor(split["test_indices"], device=device)
        base = nn.Linear(initial["weight"].shape[1], initial["weight"].shape[0])
        self.model = LoRALinear(base, rank, config.alpha).to(device)
        with torch.no_grad():
            self.model.linear.weight.copy_(initial["weight"])
            self.model.linear.bias.copy_(initial["bias"])
            self.model.lora_A.copy_(initial["A"][:rank])
            self.model.lora_B.zero_()
        self.rng = torch.Generator().manual_seed(config.current_seed + 1009 * (cid + 1))

    def train(self):
        self.model.train()
        # Reset identically every communication round, including local-only.
        optimizer = torch.optim.Adam([self.model.lora_A, self.model.lora_B],
                                     lr=self.config.lr, weight_decay=self.config.weight_decay)
        loss_total, count = 0.0, 0
        for _ in range(self.config.local_epochs):
            order = torch.randperm(len(self.train_indices), generator=self.rng).to(self.device)
            for start in range(0, len(order), self.config.batch_size):
                index = self.train_indices[order[start:start + self.config.batch_size]]
                logits = self.model(self.train_data["features"][index])
                loss = nn.functional.cross_entropy(logits, self.train_data["labels"][index])
                if not torch.isfinite(loss):
                    raise ValueError(f"non-finite training loss on client {self.client_id}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_total += float(loss.detach()) * len(index)
                count += len(index)
        return {"loss": loss_total / count, "n_samples": count}

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        correct = 0
        for start in range(0, len(self.test_indices), self.config.eval_batch_size):
            index = self.test_indices[start:start + self.config.eval_batch_size]
            correct += int((self.model(self.test_data["features"][index]).argmax(1)
                           == self.test_data["labels"][index]).sum())
        return {"accuracy": correct / len(self.test_indices), "n_samples": len(self.test_indices),
                "correct": correct}

    def get_lora_state(self):
        return {"fc": {"A": self.model.lora_A.detach().cpu().clone(),
                       "B": self.model.lora_B.detach().cpu().clone()}}

    def set_lora_state(self, state):
        with torch.no_grad():
            self.model.lora_A.copy_(state["fc"]["A"])
            self.model.lora_B.copy_(state["fc"]["B"])

    def resize_rank(self, rank):
        """Change adapter capacity while preserving the represented update."""
        rank = int(rank)
        if rank < 1 or rank == self.rank:
            if rank < 1:
                raise ValueError("rank must be positive")
            return
        old_state = self.get_lora_state()
        delta = lora_to_delta(old_state, self.alpha)["fc"]
        factors = factorize_delta(delta, rank, self.alpha, dtype=self.model.lora_A.dtype)
        base = nn.Linear(self.model.linear.in_features, self.model.linear.out_features).to(self.device)
        with torch.no_grad():
            base.weight.copy_(self.model.linear.weight)
            base.bias.copy_(self.model.linear.bias)
        from src.models.lora_resnet import LoRALinear
        replacement = LoRALinear(base, rank, self.alpha).to(self.device)
        with torch.no_grad():
            replacement.lora_A.copy_(factors["A"].to(self.device))
            replacement.lora_B.copy_(factors["B"].to(self.device))
        self.model = replacement
        self.rank = rank


def make_splits(train_labels, test_labels, config, seed):
    from src.data.cifar100_domains import get_domain_classes, partition_domain_data_dirichlet
    train_labels, test_labels = np.asarray(train_labels), np.asarray(test_labels)
    splits, assignments = {}, {}
    for domain in range(config.n_domains):
        classes = get_domain_classes(domain)
        train_indices = np.flatnonzero(np.isin(train_labels, classes))
        test_indices = np.flatnonzero(np.isin(test_labels, classes))
        rng = np.random.default_rng(seed + domain)
        rng.shuffle(test_indices)
        if config.max_test_per_domain:
            # Select equally from each class before splitting across clients.
            selected = []
            per_class, extra = divmod(config.max_test_per_domain, len(classes))
            for k, label in enumerate(classes):
                available = test_indices[test_labels[test_indices] == label]
                selected.extend(available[:per_class + (k < extra)].tolist())
            test_indices = np.asarray(selected, dtype=np.int64)
            rng.shuffle(test_indices)
        train_parts = partition_domain_data_dirichlet(train_indices, train_labels,
            config.clients_per_domain, alpha=config.dirichlet_alpha, seed=seed + domain)
        test_parts = np.array_split(test_indices, config.clients_per_domain)
        for local, (train_part, test_part) in enumerate(zip(train_parts, test_parts)):
            cid = domain * config.clients_per_domain + local
            if config.max_train_per_client and len(train_part) > config.max_train_per_client:
                train_part = rng.choice(train_part, config.max_train_per_client, replace=False).tolist()
            if not len(train_part) or not len(test_part):
                raise ValueError(f"empty data split for client {cid}; increase sample caps")
            assignments[cid] = domain
            splits[cid] = {"domain_id": domain, "train_indices": list(map(int, train_part)),
                           "test_indices": list(map(int, test_part)),
                           "train_class_counts": np.bincount(train_labels[train_part], minlength=100).tolist()}
    return splits, assignments


def split_domain_signal(splits):
    """Compute a label-distribution divergence signal without test labels."""
    counts = np.asarray([s["train_class_counts"] for _, s in sorted(splits.items())], dtype=float)
    p = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    q = counts.sum(axis=0); q = q / max(q.sum(), 1.0)
    m = 0.5 * (p + q[None, :])
    eps = 1e-12
    kl_p = np.sum(p * np.log((p + eps) / (m + eps)), axis=1)
    kl_q = np.sum(q[None, :] * np.log((q[None, :] + eps) / (m + eps)), axis=1)
    return 0.5 * (kl_p + kl_q)


def initial_parameters(feature_dim, max_rank, seed):
    # Base-head initialisation does not depend on rank; all methods share it.
    generator = torch.Generator().manual_seed(seed)
    bound = 1 / math.sqrt(feature_dim)
    return {"weight": torch.empty(100, feature_dim).uniform_(-bound, bound, generator=generator),
            "bias": torch.empty(100).uniform_(-bound, bound, generator=generator),
            "A": torch.empty(max_rank, feature_dim).uniform_(-bound, bound, generator=generator)}


@torch.no_grad()
def evaluate_protocol(clients, runner, initial, test, test_indices, consensus_rank, batch_size):
    evaluations = {c.client_id: c.evaluate() for c in clients}
    per_client = {cid: item["accuracy"] for cid, item in evaluations.items()}
    per_domain, domain_correct, domain_total = {}, {}, {}
    for client in clients:
        domain = client.domain_id
        domain_correct[domain] = domain_correct.get(domain, 0) + evaluations[client.client_id]["correct"]
        domain_total[domain] = domain_total.get(domain, 0) + evaluations[client.client_id]["n_samples"]
    per_domain = {d: domain_correct[d] / domain_total[d] for d in domain_correct}
    state = runner.consensus_state(consensus_rank)
    delta = lora_to_delta(state, runner.alpha)["fc"].to(test["features"].device)
    weight = initial["weight"].to(delta.device) + delta
    bias = initial["bias"].to(delta.device)
    consensus_correct = 0
    for start in range(0, len(test_indices), batch_size):
        indices = test_indices[start:start + batch_size]
        logits = nn.functional.linear(test["features"][indices], weight, bias)
        consensus_correct += int((logits.argmax(1) == test["labels"][indices]).sum())
    return {
        "personalized_accuracy": float(np.mean(list(per_client.values()))),
        "personalized_sample_weighted_accuracy": sum(domain_correct.values()) / sum(domain_total.values()),
        "consensus_accuracy": consensus_correct / len(test_indices),
        "per_client_accuracy": per_client, "per_domain_accuracy": per_domain,
        "accuracy_gap": max(per_domain.values()) - min(per_domain.values()),
        "worst_domain_accuracy": min(per_domain.values()),
    }


def operational_cost(method, assignments, round_idx, bridge_every, states, diagnostics, merge="delta"):
    """Exact transport counts, without claiming a simulated network speedup.

    FedAvg sends client factors to a server and its full-rank average delta
    back to clients. Hierarchy uses factors in stage one and dense exact delta
    payloads for later stages. Intermediate rank truncation would change W.
    """
    n = len(states)
    if method == "local":
        return 0, 0, "none"
    if method in {"mh", "adaptive", "weighted", "adaptive_weighted", "adaptive_rank", "adaptive_weighted_rank"}:
        return diagnostics["messages"], diagnostics["floats"], "direct neighbor factor payloads"
    sizes = [DecentralizedRunner._factor_floats(state) for state in states]
    dense_size = sum(p["A"].shape[1] * p["B"].shape[0] for p in states[0].values())
    max_rank = max(p["A"].shape[0] for state in states for p in state.values())
    padded_size = sum(max_rank * (p["A"].shape[1] + p["B"].shape[0]) for p in states[0].values())
    intermediate_size = dense_size if merge == "delta" else padded_size
    intermediate_name = "dense delta" if merge == "delta" else "zero-padded factors"
    if method == "fedavg":
        return 2 * n, sum(sizes) + n * intermediate_size, f"server gather factors; exact {intermediate_name} broadcast"
    groups = {d: [cid for cid in assignments if assignments[cid] == d] for d in set(assignments.values())}
    intra_messages = sum(len(members) * (len(members) - 1) for members in groups.values())
    ids = sorted(assignments)
    first_floats = sum((len(groups[assignments[cid]]) - 1) * sizes[i] for i, cid in enumerate(ids))
    messages = two_tier_message_cost(assignments, round_idx, bridge_every)
    bridge = round_idx % bridge_every == bridge_every - 1 and len(groups) > 1
    floats = first_floats + (intra_messages + len(groups) * (len(groups) - 1)) * intermediate_size if bridge else first_floats
    return messages, floats, f"staged exact transport: factors first, {intermediate_name} after intra mix"


def run_one(config, seed, method, train_cpu, test_cpu, feature_metadata, output):
    started = time.perf_counter()
    config.current_seed = seed
    splits, assignments = make_splits(train_cpu["labels"].numpy(), test_cpu["labels"].numpy(), config, seed)
    split_digest = hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()
    write_json(output / f"splits_seed{seed}.json", {"seed": seed, "sha256": split_digest, "clients": splits})
    ranks = {cid: config.ranks[cid % len(config.ranks)] for cid in assignments}
    initial = initial_parameters(train_cpu["features"].shape[1], max(ranks.values()), seed)
    initial_digest = tensor_digest(*initial.values())
    device = torch.device(config.device)
    train = {k: v.to(device) for k, v in train_cpu.items()}
    test = {k: v.to(device) for k, v in test_cpu.items()}
    clients = [FeatureClient(cid, assignments[cid], initial, ranks[cid], config, train, test,
                             splits[cid], device) for cid in sorted(assignments)]
    mixer = build_mixing(method, assignments, config.bridge_every, config.topology, seed, alpha=config.alpha)
    if isinstance(mixer, DomainWeightedMixer):
        mixer.signal_values = split_domain_signal(splits)
    runner_class = DecentralizedRunner if config.merge == "delta" else FactorRunner
    runner = runner_class(clients, mixer, ranks, config.alpha, config.error_feedback)
    dynamic_rank = method in {"adaptive_rank", "adaptive_weighted_rank"}
    current_ranks = dict(ranks)
    rank_history = []
    previous_losses = None
    eval_indices = torch.tensor(sorted(index for value in splits.values() for index in value["test_indices"]), device=device)
    consensus_rank = config.consensus_rank or max(ranks.values())
    window = config.bridge_every if method == "oracle" else 1
    gap = spectral_gap(window_product(mixer, 0, window)) if len(clients) > 1 else None
    record = {"schema_version": 1, "status": "running", "seed": seed, "method": method,
              "merge": config.merge, "error_feedback": config.error_feedback, "ranks": ranks,
              "rank_schedule": config.ranks, "consensus_rank": consensus_rank,
              "initial_state_sha256": initial_digest, "split_sha256": split_digest,
              "n_train_samples": sum(len(s["train_indices"]) for s in splits.values()),
              "n_test_samples": len(eval_indices), "oracle_assignments": assignments if method == "oracle" else None,
              "topology_order": topology_order(assignments, seed) if method == "mh" else None,
              "spectral_gap": gap, "spectral_window_rounds": window,
              "feature_cache_sha256": {key: feature_metadata[key]["sha256"] for key in ("train", "test")},
              "feature_cache_identity_sha256": feature_metadata.get("cache_identity_sha256"),
              "initial_metrics": evaluate_protocol(clients, runner, initial, test, eval_indices,
                                                   consensus_rank, config.eval_batch_size), "rounds": []}
    path = output / f"seed{seed}_{method}.json"
    write_json(path, record)
    for round_idx in range(config.rounds):
        round_started = time.perf_counter()
        if dynamic_rank and previous_losses is not None and round_idx >= 2:
            median_loss = float(np.median(previous_losses))
            for client in clients:
                ceiling = ranks[client.client_id]
                floor = max(1, int(np.ceil(ceiling / 2)))
                # Conservative controller: reduce only clearly easy clients;
                # restore the ceiling for clients above the round median.
                target = floor if previous_losses[client.client_id] < 0.95 * median_loss else ceiling
                client.resize_rank(target)
                current_ranks[client.client_id] = target
            runner.set_target_ranks(current_ranks)
        rank_history.append(dict(current_ranks))
        training = [client.train() for client in clients]
        previous_losses = [float(item["loss"]) for item in training]
        states = [client.get_lora_state() for client in clients]
        if method == "local":
            diagnostics = {"messages": 0, "floats": 0, "mean_tail_mass": 0.0,
                           "max_tail_mass": 0.0, "mean_residual_energy": 0.0,
                           "consensus_distance": runner._consensus_distance(states)}
        else:
            new_states, diagnostics = runner.gossip_round(round_idx, states)
            for client, state in zip(clients, new_states):
                client.set_lora_state(state)
        metrics = evaluate_protocol(clients, runner, initial, test, eval_indices,
                                    consensus_rank, config.eval_batch_size)
        messages, floats, transport = operational_cost(method, assignments, round_idx,
                                                       config.bridge_every, states, diagnostics, config.merge)
        row = {"round": round_idx + 1, **metrics,
               "mean_train_loss": float(np.mean([item["loss"] for item in training])),
               "effective_messages": diagnostics.pop("messages"),
               "effective_floats": diagnostics.pop("floats"), **diagnostics,
               "operational_messages": messages, "operational_floats": floats,
               "operational_transport": transport, "wall_seconds": time.perf_counter() - round_started}
        record["rounds"].append(row)
        record["wall_seconds"] = time.perf_counter() - started
        record["rank_history"] = rank_history
        write_json(path, record)
        print(f"seed={seed} method={method} round={round_idx + 1}/{config.rounds} "
              f"personalized={metrics['personalized_accuracy']:.4f} "
              f"consensus={metrics['consensus_accuracy']:.4f} "
              f"gap={metrics['accuracy_gap']:.4f} seconds={row['wall_seconds']:.2f}", flush=True)
    record["status"] = "complete"
    record["wall_seconds"] = time.perf_counter() - started
    record["summary"] = {"seed": seed, "method": method, "merge": config.merge,
        "rank_schedule": ":".join(map(str, config.ranks)), "error_feedback": config.error_feedback,
        "rounds": config.rounds, **{k: record["rounds"][-1][k] for k in SUMMARY_FIELDS if k in record["rounds"][-1]},
        **{f"total_{name}": sum(row[name] for row in record["rounds"])
           for name in ("effective_messages", "effective_floats", "operational_messages", "operational_floats")},
        "wall_seconds": record["wall_seconds"], "initial_state_sha256": initial_digest, "split_sha256": split_digest}
    write_json(path, record)
    return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing run files are refused")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--feature-cache", type=Path, default=Path("data/features"))
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=["local", "fedavg", "mh", "oracle"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--ranks", nargs="+", type=int, default=[8], help="Rank cycle across client ids")
    parser.add_argument("--consensus-rank", type=int, default=0, help="0 uses the largest client rank")
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--merge", choices=["delta", "factor_zero_pad"], default="delta")
    parser.add_argument("--error-feedback", action="store_true")
    parser.add_argument("--n-domains", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--clients-per-domain", type=int, default=3)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--bridge-every", type=int, default=5)
    parser.add_argument("--topology", choices=["ring", "fully_connected", "star", "path"], default="ring")
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--max-train-per-client", type=int, default=0, help="0 uses all data; positive values mark a subset run")
    parser.add_argument("--max-test-per-domain", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prepare-only", action="store_true", help="Build/verify feature cache and stop before training")
    args = parser.parse_args(argv)
    for name in ("rounds", "clients_per_domain", "bridge_every", "local_epochs", "batch_size", "eval_batch_size",
                 "feature_batch_size", "torch_threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if any(rank < 1 for rank in args.ranks) or args.consensus_rank < 0:
        parser.error("ranks must be positive and consensus rank nonnegative")
    for name in ("alpha", "lr", "dirichlet_alpha"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight decay must be finite and nonnegative")
    if args.max_train_per_client < 0 or args.max_test_per_domain < 0 or args.num_workers < 0:
        parser.error("sample caps and num workers must be nonnegative")
    if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds):
        parser.error("methods and seeds must be unique")
    if any(seed < 0 or seed > 2 ** 32 - 1 for seed in args.seeds):
        parser.error("seeds must be in [0, 2**32 - 1]")
    if args.error_feedback and args.merge != "delta":
        parser.error("error feedback requires --merge delta")
    if any(name in args.methods for name in ("adaptive", "adaptive_weighted", "adaptive_rank", "adaptive_weighted_rank")) and args.merge != "delta":
        parser.error("adaptive discovery requires --merge delta")
    return args


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
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(config).items()}
    def git(*args):
        completed = subprocess.run(["git", *args], cwd=Path(__file__).resolve().parents[2],
                                   capture_output=True, text=True, check=False)
        return completed.stdout.strip() if completed.returncode == 0 else "unavailable"
    classification = "smoke" if config.rounds < 5 else ("subset_benchmark" if config.max_train_per_client or config.max_test_per_domain else "full_data_benchmark")
    manifest = {"schema_version": 1, "status": "preparing_features", "run_classification": classification,
        "config": configuration, "started_unix": time.time(),
        "git": {"commit": git("rev-parse", "HEAD"), "working_tree_status": git("status", "--short")},
        "source": source_provenance(),
        "invocation": {"executable": sys.executable,
                       "argv": sys.argv if argv is None else [sys.argv[0], *argv],
                       "cwd": str(Path.cwd())},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "cuda": torch.version.cuda,
                        "packages": package_versions(), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "device": config.device, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None},
        "protocol": {"representation": "cached frozen eval-mode ResNet-18 features; no augmentation",
          "objective": "uniform client mean; FedAvg is client-uniform, not sample-weighted",
          "head": "shared random frozen 100-class linear base plus trainable LoRA; common initialization per seed",
          "optimizer": "Adam reset every round for every method, including local; paired client minibatch RNGs",
          "personalized_accuracy": "arithmetic mean of client adapter accuracy on disjoint client test shards",
          "personalized_sample_weighted_accuracy": "total correct across client test shards / total samples",
          "per_domain_accuracy": "sample-weighted accuracy within each domain",
          "consensus_accuracy": "uniform mean effective adapter, SVD at consensus_rank, on union of test shards",
          "oracle": "known CIFAR superclass domain labels; no learned discovery or transfer weights",
          "flat_topology": "seeded permutation of client IDs independent of domain, matrix reordered to client order",
          "local": "no communication and no SVD reparameterization between local rounds",
          "communication": "simulated scalar float counts; multiply by 4 for fp32 bytes; excludes setup/evaluation",
          "operational_transport": "MH factors; FedAvg factors up/dense delta down; hierarchy factors first/dense later",
          "factor_baseline": "naive zero-padded factor averaging; merge error is not SVD tail mass",
          "adaptive_rank": "loss-adaptive capacity policy: two-round warmup, then half-ceiling for clients below 95% of the previous round median loss; rank changes preserve effective Delta-W",
          "domain_weighting": "bounded update-norm factors (blend 0.10, deviation bound 0.15) applied through symmetric Sinkhorn reweighting of the gossip matrix"},
        "completed_runs": []}
    write_json(config.output / "manifest.json", manifest)
    try:
        train, test, metadata = load_features(config.data_dir, config.feature_cache, torch.device(config.device),
            config.image_size, config.feature_batch_size, config.num_workers)
        manifest["feature_cache"] = metadata
        manifest["status"] = "prepared" if config.prepare_only else "running"
        write_json(config.output / "manifest.json", manifest)
        if config.prepare_only:
            return manifest
        with (config.output / "summary.csv").open("w", newline="") as summary_file:
            writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
            writer.writeheader()
            summary_file.flush()
            for seed in config.seeds:
                for method in config.methods:
                    record = run_one(config, seed, method, train, test, metadata, config.output)
                    with (config.output / "results.jsonl").open("a") as handle:
                        handle.write(json.dumps(record, allow_nan=False) + "\n")
                    writer.writerow(record["summary"])
                    summary_file.flush()
                    manifest["completed_runs"].append({"seed": seed, "method": method, "file": f"seed{seed}_{method}.json"})
                    write_json(config.output / "manifest.json", manifest)
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        manifest["wall_seconds"] = time.time() - manifest["started_unix"]
        write_json(config.output / "manifest.json", manifest)
    return manifest
