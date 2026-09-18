"""Explicit peer payload simulation for quantity-skew transformer LoRA.

State schema: ``{'adapter': {layer: {'A': tensor, 'B': tensor}},
                 'head': {parameter_name: tensor}}``.

This is synchronous single-process simulation, not RPC, isolation, secure
aggregation, or a privacy mechanism. Every nonlocal state used in a merge is
serialized onto a declared graph edge and decoded at its receiver. Traffic
includes raw tensor bytes and a JSON metadata header carrying source IDs,
weights, ranks, shapes, dtypes, and parameter names. No network framing beyond
the explicit eight-byte header length is assumed.
"""

from collections.abc import Mapping
import hashlib
import json
import math
import numbers
import struct

import numpy as np
import torch


_DTYPES = {str(dtype).split(".")[-1]: dtype for dtype in
           (torch.float16, torch.bfloat16, torch.float32, torch.float64)}


def _normalized(weights, size):
    values = np.asarray(weights, dtype=np.float64)
    if (values.shape != (size,) or not np.isfinite(values).all()
            or np.any(values < 0) or not np.any(values > 0)):
        raise ValueError("weights must have one finite nonnegative value per state and positive total")
    values = values / values.max()
    return values / values.sum()


def _validate_state(state):
    if not isinstance(state, Mapping) or set(state) != {"adapter", "head"}:
        raise ValueError("peer state must contain exactly adapter and head mappings")
    adapter, head = state["adapter"], state["head"]
    if not isinstance(adapter, Mapping) or not adapter or not isinstance(head, Mapping):
        raise ValueError("adapter must be nonempty and head must be a mapping")
    if any(not isinstance(name, str) for name in (*adapter, *head)):
        raise ValueError("layer and head parameter names must be strings")
    for name, factors in adapter.items():
        if not isinstance(factors, Mapping) or set(factors) != {"A", "B"}:
            raise ValueError(f"adapter layer {name} must contain A and B")
        a, b = factors["A"], factors["B"]
        if (not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor)
                or a.ndim != 2 or b.ndim != 2 or min(*a.shape, *b.shape) < 1
                or a.shape[0] != b.shape[1] or a.dtype != b.dtype or a.device != b.device):
            raise ValueError(f"invalid LoRA factor shapes, dtypes, or devices for {name}")
    for tensor in _tensors(state):
        if (not isinstance(tensor, torch.Tensor) or tensor.dtype not in _DTYPES.values()
                or not torch.isfinite(tensor).all()):
            raise ValueError("peer tensors must be finite supported floating-point tensors")


def _tensors(state):
    for name in sorted(state["adapter"]):
        yield state["adapter"][name]["A"]
        yield state["adapter"][name]["B"]
    for name in sorted(state["head"]):
        yield state["head"][name]


def _tensor_bytes(state):
    return sum(tensor.numel() * tensor.element_size() for tensor in _tensors(state))


def _validate_states(states):
    states = list(states)
    if not states:
        raise ValueError("at least one peer state is required")
    for state in states:
        _validate_state(state)
    reference = states[0]
    for state in states[1:]:
        if set(state["adapter"]) != set(reference["adapter"]) or set(state["head"]) != set(reference["head"]):
            raise ValueError("peer states must have identical layer and head parameter names")
        for name, factors in reference["adapter"].items():
            other = state["adapter"][name]
            if factors["A"].shape[1] != other["A"].shape[1] or factors["B"].shape[0] != other["B"].shape[0]:
                raise ValueError("effective adapter dimensions must agree across peers")
        if any(state["head"][name].shape != tensor.shape for name, tensor in reference["head"].items()):
            raise ValueError("head parameter shapes must agree across peers")
    return states


def _average_tensors(tensors, weights):
    reference = tensors[0]
    dtype = torch.float64 if any(t.dtype == torch.float64 for t in tensors) else torch.float32
    result = torch.zeros_like(reference, dtype=dtype)
    for tensor, weight in zip(tensors, weights):
        result.add_(tensor.detach().to(device=reference.device, dtype=dtype), alpha=float(weight))
    # Preserve double precision if any sender used it; otherwise preserve the
    # reference storage dtype while doing half-precision sums in float32.
    return result.to(dtype=torch.float64 if dtype == torch.float64 else reference.dtype)


def mix_heads(states, weights):
    """Convex average of trainable classifier parameters, including biases."""
    states = _validate_states(states)
    weights = _normalized(weights, len(states))
    return {name: _average_tensors([state["head"][name] for state in states], weights)
            for name in sorted(states[0]["head"])}


def factor_mix(states, weights):
    """Paper-style equal-rank factor average plus classifier-head average.

    This deliberately averages A and B separately. It reproduces that baseline
    rule and is not described as equivalent to averaging effective products.
    """
    states = _validate_states(states)
    weights = _normalized(weights, len(states))
    adapter = {}
    for layer in sorted(states[0]["adapter"]):
        reference = states[0]["adapter"][layer]
        if any(state["adapter"][layer]["A"].shape != reference["A"].shape for state in states):
            raise ValueError("factor_mix requires equal ranks for each layer across peers")
        adapter[layer] = {factor: _average_tensors(
            [state["adapter"][layer][factor] for state in states], weights) for factor in ("A", "B")}
    return {"adapter": adapter, "head": mix_heads(states, weights)}


def _encode(records):
    """Serialize a list of peer-owned records; byte counts equal wire length."""
    header, buffers = {"schema_version": 1, "records": [], "tensors": []}, []

    def add_tensor(tensor):
        cpu = tensor.detach().cpu().contiguous()
        raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
        index = len(buffers)
        buffers.append(raw)
        header["tensors"].append({"shape": list(cpu.shape),
                                   "dtype": str(cpu.dtype).split(".")[-1], "nbytes": len(raw)})
        return index

    for record in records:
        state = record["state"]
        item = {"client_id": int(record["client_id"]), "weight": float(record["weight"]),
                "adapter": {}, "head": {}}
        for layer in sorted(state["adapter"]):
            factors = state["adapter"][layer]
            item["adapter"][layer] = {"rank": factors["A"].shape[0],
                                      "A": add_tensor(factors["A"]), "B": add_tensor(factors["B"])}
        for name in sorted(state["head"]):
            item["head"][name] = add_tensor(state["head"][name])
        header["records"].append(item)
    metadata = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    wire = struct.pack("<Q", len(metadata)) + metadata + b"".join(buffers)
    return wire, sum(map(len, buffers))


def _decode(wire):
    header_size = struct.unpack("<Q", wire[:8])[0]
    header = json.loads(wire[8:8 + header_size])
    offset, tensors = 8 + header_size, []
    for spec in header["tensors"]:
        size, dtype, shape = spec["nbytes"], _DTYPES[spec["dtype"]], spec["shape"]
        if math.prod(shape) * torch.empty((), dtype=dtype).element_size() != size:
            raise ValueError("serialized tensor size disagrees with its metadata")
        if size:
            tensor = torch.frombuffer(bytearray(wire[offset:offset + size]), dtype=dtype).reshape(shape).clone()
        else:
            tensor = torch.empty(shape, dtype=dtype)
        tensors.append(tensor)
        offset += size
    if offset != len(wire):
        raise ValueError("serialized payload length disagrees with its metadata")
    records = []
    for item in header["records"]:
        state = {"adapter": {name: {key: tensors[spec[key]] for key in ("A", "B")}
                             for name, spec in item["adapter"].items()},
                 "head": {name: tensors[index] for name, index in item["head"].items()}}
        if any(spec["rank"] != state["adapter"][name]["A"].shape[0]
               for name, spec in item["adapter"].items()):
            raise ValueError("serialized rank disagrees with factor shape")
        records.append({"client_id": item["client_id"], "weight": item["weight"], "state": state})
    return records


def _transfer(records, sender, receiver, phase):
    wire, tensor_bytes = _encode(records)
    received = _decode(wire)
    ledger = {"sender": int(sender), "receiver": int(receiver), "phase": phase,
              "source_client_ids": [int(record["client_id"]) for record in records],
              "source_weights": [float(record["weight"]) for record in records],
              "record_count": len(records), "tensor_bytes": tensor_bytes,
              "metadata_bytes": len(wire) - tensor_bytes, "payload_bytes": len(wire),
              "payload_sha256": hashlib.sha256(wire).hexdigest()}
    return received, ledger


def _graph(states, neighbors):
    if (not isinstance(states, Mapping) or not isinstance(neighbors, Mapping)
            or not states or set(states) != set(neighbors)):
        raise ValueError("states and neighbors must identify the same nonempty peer set")
    if any(isinstance(cid, (bool, np.bool_)) or not isinstance(cid, numbers.Integral) for cid in states):
        raise ValueError("client IDs must be integers")
    ids = sorted(states)
    _validate_states([states[cid] for cid in ids])
    for cid in ids:
        peers = neighbors[cid]
        if len(peers) != len(set(peers)) or cid in peers:
            raise ValueError("neighbors must not contain duplicates or self edges")
        if any(peer not in states or cid not in neighbors[peer] for peer in peers):
            raise ValueError("neighbors must be a symmetric graph over the peer set")
    return ids


def _merge(records, mode, target_rank, alpha):
    states = [record["state"] for record in records]
    weights = [record["weight"] for record in records]
    if target_rank is not None and (isinstance(target_rank, (bool, np.bool_))
                                   or not isinstance(target_rank, numbers.Integral) or target_rank < 1):
        raise ValueError("target_rank must be a positive integer")
    if mode == "factor":
        merged = factor_mix(states, weights)
        if target_rank is not None and any(factors["A"].shape[0] != target_rank
                                           for factors in merged["adapter"].values()):
            raise ValueError("factor mode cannot change the common adapter rank")
        return merged, {"method": "separate_factor_average", "normalized_weights": _normalized(weights, len(states)).tolist()}
    if mode != "effective":
        raise ValueError("mode must be factor or effective")
    if isinstance(target_rank, (bool, np.bool_)) or not isinstance(target_rank, numbers.Integral) or target_rank < 1:
        raise ValueError("effective mode needs a positive integer target_rank")
    from src.federated.compact_merge import merge_compact
    adapter, diagnostics = merge_compact([state["adapter"] for state in states], weights,
                                         int(target_rank), alpha)
    return {"adapter": adapter, "head": mix_heads(states, weights)}, diagnostics


def _to_reference_devices(state, reference):
    return {"adapter": {layer: {factor: tensor.to(reference["adapter"][layer][factor].device)
                                 for factor, tensor in factors.items()}
                         for layer, factors in state["adapter"].items()},
            "head": {name: tensor.to(reference["head"][name].device)
                     for name, tensor in state["head"].items()}}


def _ledger(transfers, protocol):
    result = {"protocol": protocol, "execution": "single-process explicit serialized payload simulation",
              "messages": len(transfers), "bytes": sum(row["payload_bytes"] for row in transfers),
              "tensor_bytes": sum(row["tensor_bytes"] for row in transfers),
              "metadata_bytes": sum(row["metadata_bytes"] for row in transfers),
              "max_wire_payload_bytes": max((row["payload_bytes"] for row in transfers), default=0),
              "transfers": transfers,
              "byte_scope": "Tensor values plus JSON IDs, weights, ranks, shapes, dtypes, names and eight-byte header length; excludes network framing, graph setup, optimizer/base-model state."}
    for phase in ("gather", "dissemination", "gossip"):
        rows = [row for row in transfers if row["phase"] == phase]
        result[f"{phase}_messages"] = len(rows)
        result[f"{phase}_bytes"] = sum(row["payload_bytes"] for row in rows)
        result[f"{phase}_tensor_bytes"] = sum(row["tensor_bytes"] for row in rows)
    return result


def tree_assemble(states_by_id, weights_by_id, neighbors, root_id=0, *, mode="effective",
                  target_rank=None, alpha=16.0, disseminate=True):
    """Gather serialized records on a BFS tree, merge at root, then broadcast.

    Each child forwards its own and already received subtree records. The root
    merges only records delivered by that tree plus its own local state. It
    explicitly stores all received factor/head records; this is not a bounded
    memory streaming reduction. Return ``(assembled_peer_state, ledger)``.
    Dissemination is a counted delivery simulation; it does not mutate clients.
    """
    ids = _graph(states_by_id, neighbors)
    if (not isinstance(weights_by_id, Mapping) or set(weights_by_id) != set(ids)
            or isinstance(root_id, (bool, np.bool_)) or root_id not in states_by_id):
        raise ValueError("weights must cover exactly the peers and root_id must be a peer")
    normalized = _normalized([weights_by_id[cid] for cid in ids], len(ids))
    if any(float(weights_by_id[cid]) <= 0 for cid in ids):
        raise ValueError("assembly weights must be strictly positive")
    parent, order, depth = {root_id: None}, [root_id], {root_id: 0}
    for cid in order:
        for peer in sorted(neighbors[cid]):
            if peer not in parent:
                parent[peer], depth[peer] = cid, depth[cid] + 1
                order.append(peer)
    if len(order) != len(ids):
        raise ValueError("assembly graph must be connected")
    packets = {cid: [{"client_id": int(cid), "weight": float(weights_by_id[cid]),
                      "state": states_by_id[cid]}] for cid in ids}
    peak = {cid: _tensor_bytes(states_by_id[cid]) for cid in ids}
    transfers = []
    for sender in reversed(order[1:]):
        receiver = parent[sender]
        received, row = _transfer(packets[sender], sender, receiver, "gather")
        packets[receiver].extend(received)
        peak[receiver] = max(peak[receiver], sum(_tensor_bytes(record["state"]) for record in packets[receiver]))
        packets[sender] = []
        transfers.append(row)
    root_records = sorted(packets[root_id], key=lambda record: record["client_id"])
    # Serialization delivers CPU tensors; copy root's local record to the same
    # device before merging. This copies only its own state, not other peers.
    root_records = [{**record, "state": _decode(_encode([record])[0])[0]["state"]}
                    if record["client_id"] == root_id else record for record in root_records]
    merged, diagnostics = _merge(root_records, mode, target_rank, alpha)
    if disseminate:
        forwarded = {root_id: [{"client_id": int(root_id), "weight": 1.0, "state": merged}]}
        for receiver in order[1:]:
            sender = parent[receiver]
            received, row = _transfer(forwarded[sender], sender, receiver, "dissemination")
            forwarded[receiver] = received
            transfers.append(row)
    ledger = _ledger(transfers, "neighbor-only BFS factor-record gather, root assembly, optional dissemination")
    ledger.update({"root_id": int(root_id), "mode": mode, "target_rank": None if target_rank is None else int(target_rank),
                   "client_order": list(map(int, ids)), "tree_parent": {int(k): None if v is None else int(v) for k, v in parent.items()},
                   "tree_depth": {int(k): int(v) for k, v in depth.items()},
                   "normalized_weights": {int(cid): float(weight) for cid, weight in zip(ids, normalized)},
                   "source_weights": {int(cid): float(weights_by_id[cid]) for cid in ids},
                   "root_assembly_input_tensor_bytes": sum(_tensor_bytes(record["state"]) for record in root_records),
                   "assembled_tensor_bytes": _tensor_bytes(merged),
                   "peak_gather_record_tensor_bytes_by_peer": {int(cid): int(size) for cid, size in peak.items()},
                   "memory_scope": "Logical per-peer gather record tensors; root explicitly stores every source record. Excludes original training model, wire buffers, Python objects, optimizer state and merge/SVD workspace; output and largest wire buffer are reported separately.",
                   "merge_diagnostics": diagnostics, "disseminated": bool(disseminate)})
    return _to_reference_devices(merged, states_by_id[root_id]), ledger


def neighbor_round(states_by_id, matrix, neighbors, ranks, alpha, mode="effective"):
    """One synchronous graph-restricted factor/compact merge from old states.

    Matrix rows/columns follow sorted client IDs. Every positive off-diagonal
    contribution is sent on a declared graph edge, carrying that receiver's
    scalar coefficient. Self contributions remain local. Return new states
    without mutating the supplied states or publishing them during the round.
    """
    ids = _graph(states_by_id, neighbors)
    matrix = np.asarray(matrix, dtype=np.float64)
    if (matrix.shape != (len(ids), len(ids)) or not np.isfinite(matrix).all()
            or np.any(matrix < 0) or not np.allclose(matrix.sum(1), 1, atol=1e-12, rtol=0)):
        raise ValueError("matrix must be finite nonnegative row-stochastic over sorted client IDs")
    if set(ranks) != set(ids):
        raise ValueError("ranks must identify exactly the peer set")
    for i, receiver in enumerate(ids):
        for j, sender in enumerate(ids):
            if sender != receiver and matrix[i, j] > 0 and sender not in neighbors[receiver]:
                raise ValueError("matrix uses an undeclared non-neighbor edge")
    inboxes, transfers = {}, []
    for i, receiver in enumerate(ids):
        inboxes[receiver] = []
        for j, sender in enumerate(ids):
            if matrix[i, j] == 0:
                continue
            record = {"client_id": int(sender), "weight": float(matrix[i, j]), "state": states_by_id[sender]}
            if sender == receiver:
                # Local data are copied into CPU simulation buffers but do not
                # create a fictitious network message.
                received = _decode(_encode([record])[0])
            else:
                received, row = _transfer([record], sender, receiver, "gossip")
                transfers.append(row)
            inboxes[receiver].extend(received)
    new_states, diagnostics = {}, {}
    for cid in ids:
        merged, diagnostics[cid] = _merge(inboxes[cid], mode, ranks[cid], alpha)
        new_states[cid] = _to_reference_devices(merged, states_by_id[cid])
    ledger = _ledger(transfers, "synchronous graph-only serialized neighbor state exchange")
    ledger.update({"client_order": list(map(int, ids)), "mode": mode,
                   "merge_diagnostics": diagnostics,
                   "inbox_tensor_bytes_by_peer": {int(cid): sum(_tensor_bytes(record["state"]) for record in inboxes[cid]) for cid in ids},
                   "memory_scope": "Logical decoded per-peer inbox tensors; excludes model/optimizer state, wire buffers, merge workspace, and the single-process coordinator's copies."})
    return new_states, ledger


__all__ = ["factor_mix", "mix_heads", "tree_assemble", "neighbor_round"]
