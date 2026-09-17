"""Neighbor-only final assembly at a predeclared deployment peer.

This is a deterministic transport simulation, not a distributed RPC runtime.
Each peer starts with its own scaled LoRA update and a positive contribution
weight. A breadth-first spanning tree of the supplied peer graph routes one
weighted dense update numerator and one scalar mass from every non-root peer
to its parent. The deployment peer divides by the received total mass and
refactorizes once. No intermediate rank truncation or all-client averaging
operator is hidden in the reduction.

Dense payloads and dense temporary storage are necessary for this exact
reduction; they are explicitly counted. The implementation does not provide
privacy: the root sees the assembled update and a parent sees its child's
subtree aggregate. Tree/weight setup and model dissemination after assembly
are outside the returned transport cost.
"""

import math
from collections.abc import Mapping
from numbers import Real

import numpy as np
import torch

from src.federated.merge import _check_alpha, factorize_delta, lora_to_delta
from src.federated.mixing import _validate


def tree_weighted_assembly(states, weights, neighbors, root_id, target_rank, alpha):
    """Return ``(root_lora_state, diagnostics)`` after a peer-tree reduction.

    ``states`` and ``weights`` map client identifiers to adapter states and
    positive, not necessarily normalized contribution weights. ``neighbors``
    is the symmetric connected training graph. The caller must select
    ``root_id`` and ``target_rank`` independently of test-set performance.

    Transport values include fp32/fp64 dense tensors and a float64 scalar mass
    per tree edge. ``peak_peer_dense_bytes`` counts an accumulator plus one
    incoming tensor buffer, excluding adapters, SVD and diagnostic workspace.
    All peers are simulated in one process, whose actual memory is larger.
    """
    _check_alpha(alpha)
    if not states:
        raise ValueError("at least one peer state is required")
    if set(states) != set(weights) or set(states) != set(neighbors):
        raise ValueError("states, weights, and neighbors must name the same peers")
    if root_id not in states:
        raise ValueError("root_id must name an existing peer")
    if isinstance(target_rank, bool) or int(target_rank) != target_rank or target_rank < 1:
        raise ValueError("target_rank must be a positive integer")
    _validate(neighbors)
    masses = {cid: float(weights[cid]) for cid in states}
    if any(not math.isfinite(w) or w <= 0 for w in masses.values()):
        raise ValueError("assembly weights must be finite and strictly positive")
    try:
        total_weight = math.fsum(masses.values())
    except OverflowError as error:
        raise ValueError("assembly weights must have a finite total") from error
    if not math.isfinite(total_weight):
        raise ValueError("assembly weights must have a finite total")

    # BFS only follows the supplied communication graph. Reverse BFS order is
    # a postorder: descendants have sent their numerators before parents send.
    parent, order = {root_id: None}, [root_id]
    for cid in order:
        for peer in neighbors[cid]:
            if peer not in parent:
                parent[peer] = cid
                order.append(peer)
    if len(order) != len(states):
        raise ValueError("assembly graph must be connected")

    layers = list(states[root_id])
    if not layers:
        raise ValueError("peer adapter states must have at least one layer")
    reference = None
    accumulators = {}
    for cid in order:
        if set(states[cid]) != set(layers):
            raise ValueError("peer adapter states must have the same layers")
        delta = lora_to_delta(states[cid], alpha)
        descriptor = {name: (tuple(delta[name].shape), delta[name].dtype,
                             delta[name].device) for name in layers}
        if reference is None:
            reference = descriptor
        elif descriptor != reference:
            raise ValueError("peer updates must have matching shapes, dtypes, and devices")
        if any(not torch.isfinite(delta[name]).all() for name in layers):
            raise ValueError("peer updates must be finite")
        accumulators[cid] = {name: delta[name] * masses[cid] for name in layers}
    del delta
    dense_floats = sum(value.numel() for value in accumulators[root_id].values())
    dense_bytes = sum(value.numel() * value.element_size()
                      for value in accumulators[root_id].values())

    messages = []
    for cid in reversed(order[1:]):
        receiver = parent[cid]
        payload = accumulators.pop(cid)
        for layer in layers:
            accumulators[receiver][layer].add_(payload[layer])
        masses[receiver] += masses[cid]
        messages.append({
            "sender": cid, "receiver": receiver,
            "subtree_weight": masses[cid],
            "dense_floats": dense_floats, "scalar_metadata_values": 1,
            "payload_bytes": dense_bytes + 8,
        })
        del payload

    numerator = accumulators[root_id]
    if any(not torch.isfinite(value).all() for value in numerator.values()):
        raise ValueError("weighted assembly overflowed; rescale contribution weights")
    averaged = {name: value.div_(masses[root_id]) for name, value in numerator.items()}
    root_state = {
        name: factorize_delta(value, int(target_rank), alpha,
                              dtype=states[root_id][name]["A"].dtype)
        for name, value in averaged.items()
    }
    reconstructed = lora_to_delta(root_state, alpha)
    total_energy = sum(float(torch.sum(value ** 2)) for value in averaged.values())
    residual_energy = sum(float(torch.sum((averaged[name] - reconstructed[name]) ** 2))
                          for name in layers)
    diagnostics = {
        "protocol": "neighbor-only spanning-tree weighted dense reduction",
        "execution": "single-process transport simulation",
        "root_id": root_id, "target_rank": int(target_rank),
        "n_peers": len(states), "total_weight": masses[root_id],
        "normalized_weights": {cid: float(weights[cid]) / total_weight for cid in states},
        "messages": len(messages), "dense_floats": dense_floats * len(messages),
        "scalar_metadata_values": len(messages),
        "floats": (dense_floats + 1) * len(messages),
        "bytes": (dense_bytes + 8) * len(messages),
        "dense_floats_per_payload": dense_floats,
        "peak_peer_dense_bytes": dense_bytes * (2 if messages else 1),
        "memory_scope": "transport accumulator plus incoming dense buffer; excludes adapters, SVD and diagnostic workspace",
        "transport_scope": "final reduction only; excludes tree/weight setup and later model dissemination",
        "tree_edges": messages,
        "truncation_energy": residual_energy,
        "relative_truncation_energy": residual_energy / total_energy if total_energy else 0.0,
    }
    return root_state, diagnostics


def _numeric_payload_size(payload):
    """Number of numeric scalars under an explicit float64 transport model."""
    if isinstance(payload, torch.Tensor):
        if payload.is_complex() or not torch.isfinite(payload).all():
            raise ValueError("allgather payloads must contain finite real numbers")
        return payload.numel()
    if isinstance(payload, np.ndarray):
        if not np.issubdtype(payload.dtype, np.number) or np.iscomplexobj(payload) or not np.isfinite(payload).all():
            raise ValueError("allgather payloads must contain finite real numbers")
        return int(payload.size)
    if isinstance(payload, Real):
        if not math.isfinite(float(payload)):
            raise ValueError("allgather payloads must contain finite real numbers")
        return 1
    if isinstance(payload, Mapping):
        return sum(_numeric_payload_size(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return sum(_numeric_payload_size(value) for value in payload)
    raise ValueError("allgather payloads must be numeric scalars, arrays, lists, or mappings")


def tree_allgather(payloads, neighbors, root_id):
    """Simulate graph-restricted gather/broadcast of per-peer numeric records.

    Return ``({peer: {source_peer: payload}}, diagnostics)``. Payloads may be
    nested numeric mappings/lists, NumPy arrays or tensors. Every result view
    must be treated as read-only: the simulator shares payload references,
    while its ledger charges the full numerical contents on every tree edge.
    Diagnostic fields are JSON serializable and contain no payload contents.

    This deliberately simple control plane exposes all participating records
    to all peers. It is neither secure aggregation nor a privacy mechanism.
    Numeric fields use an explicit eight-byte scalar transport model, with
    one eight-byte source identifier per record. Packet framing and graph
    setup are excluded. A real implementation could reduce sufficient
    statistics instead; such an optimization is not simulated here.
    """
    if not payloads or set(payloads) != set(neighbors):
        raise ValueError("payloads and neighbors must name the same nonempty peers")
    if root_id not in payloads:
        raise ValueError("root_id must name an existing peer")
    _validate(neighbors)
    parent, order = {root_id: None}, [root_id]
    for cid in order:
        for peer in neighbors[cid]:
            if peer not in parent:
                parent[peer] = cid
                order.append(peer)
    if len(order) != len(payloads):
        raise ValueError("allgather graph must be connected")
    sizes = {cid: _numeric_payload_size(value) for cid, value in payloads.items()}
    gathered = {cid: {cid: payloads[cid]} for cid in order}
    ledger = []

    def log_transfer(sender, receiver, records, phase):
        values = sum(sizes[cid] for cid in records)
        ledger.append({
            "sender": sender, "receiver": receiver, "phase": phase,
            "records": len(records), "numeric_values": values,
            "source_identifier_values": len(records),
            "payload_bytes": 8 * (values + len(records)),
        })

    for cid in reversed(order[1:]):
        receiver = parent[cid]
        log_transfer(cid, receiver, gathered[cid], "gather")
        gathered[receiver].update(gathered.pop(cid))
    views = {root_id: gathered[root_id]}
    for cid in order[1:]:
        sender = parent[cid]
        log_transfer(sender, cid, views[sender], "broadcast")
        views[cid] = dict(views[sender])
    full_view_bytes = 8 * (sum(sizes.values()) + len(payloads))
    diagnostics = {
        "protocol": "neighbor-only spanning-tree numeric allgather",
        "execution": "single-process transport simulation; read-only payload views share physical memory",
        "root_id": root_id, "n_peers": len(payloads),
        "messages": len(ledger),
        "numeric_values": sum(message["numeric_values"] for message in ledger),
        "source_identifier_values": sum(message["source_identifier_values"] for message in ledger),
        "bytes": sum(message["payload_bytes"] for message in ledger),
        "full_view_bytes_per_peer": full_view_bytes,
        "peak_peer_payload_bytes_upper_bound": full_view_bytes * (2 if ledger else 1),
        "transport_scope": "float64 numerical payloads and int64 source IDs; excludes framing and graph setup",
        "privacy_scope": "every peer receives all source records; no privacy protection",
        "tree_edges": ledger,
    }
    return views, diagnostics
