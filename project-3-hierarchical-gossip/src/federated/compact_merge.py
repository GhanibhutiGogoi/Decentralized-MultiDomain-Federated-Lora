"""Effective-update LoRA merging through QR and a compact core SVD.

For normalized nonnegative weights w_i and client ranks r_i, form
L = [sqrt(w_i * alpha/r_i) B_i]_i and
R = [sqrt(w_i * alpha/r_i) A_i.T]_i.  The desired update is L R.T.
Thin QR gives L=Q_L R_L and R=Q_R R_R, so its nonzero singular
values come from the small core R_L R_R.T.  We never explicitly form
the out_features-by-in_features effective update.  If the sum of client
ranks reaches the feature dimensions, the core itself can become dense;
diagnostics expose that condition rather than claiming universal savings.

This is an aggregation operation, not a differentiable training layer.
The returned factors use the existing convention (alpha/target_rank) B A.
"""

from collections.abc import Mapping
import math
from numbers import Integral

import torch


def _weights(weights, count):
    values = list(weights)
    if len(values) != count:
        raise ValueError("weights must contain one value per state")
    values = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("weights must be finite and nonnegative")
    # Scaling first avoids overflowing the sum for valid, large weights.
    largest = max(values)
    if largest == 0:
        raise ValueError("weights must have a positive total")
    scaled = [value / largest for value in values]
    total = math.fsum(scaled)
    return [value / total for value in scaled]


def _validate_layer(states, layer):
    shape = None
    device = None
    dtype = torch.float32
    for index, state in enumerate(states):
        params = state[layer]
        if not isinstance(params, Mapping) or "A" not in params or "B" not in params:
            raise ValueError(f"state {index}, layer {layer}: expected A and B factors")
        a, b = params["A"], params["B"]
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise ValueError(f"state {index}, layer {layer}: factors must be tensors")
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise ValueError(f"state {index}, layer {layer}: incompatible factor shapes")
        if a.shape[1] < 1 or b.shape[0] < 1:
            raise ValueError(f"state {index}, layer {layer}: feature dimensions must be positive")
        if a.dtype not in (torch.float32, torch.float64) or b.dtype != a.dtype:
            raise ValueError(f"state {index}, layer {layer}: matching fp32/fp64 factors required")
        if a.device != b.device or (device is not None and a.device != device):
            raise ValueError(f"state {index}, layer {layer}: factors must share a device")
        current_shape = (b.shape[0], a.shape[1])
        if shape is not None and current_shape != shape:
            raise ValueError(f"state {index}, layer {layer}: feature shapes differ across states")
        if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
            raise ValueError(f"state {index}, layer {layer}: factors must be finite")
        shape, device = current_shape, a.device
        if a.dtype == torch.float64:
            dtype = torch.float64
    return shape, device, dtype


@torch.no_grad()
def merge_compact(states, weights, target_rank, alpha):
    """Return ``(merged_state, diagnostics)`` without constructing dense deltas.

    ``states`` contain ``{layer: {'A': [r,in], 'B': [out,r]}}``. Ranks may
    differ across clients/layers, including rank zero (a zero contribution).
    A common finite positive ``alpha`` applies to every input and the output;
    each input still has its own alpha/r scaling. Weights are normalized.
    ``target_rank`` is a nonnegative integer; zero returns empty A/B factors
    and records all effective-update energy as discarded. Ranks exceeding
    the available core dimensions are padded with zeros. Mixed fp32/fp64
    clients are promoted per layer to fp64. Inputs are not modified.

    Workspace diagnostics are structural tensor-size estimates, not measured
    memory peaks. They exclude model/optimizer/input storage, allocator
    caching, QR/SVD implementation workspace, and backend temporary buffers.
    Layers are processed sequentially; output factors accumulate in memory.
    """
    states = list(states)
    if not states:
        raise ValueError("merge_compact requires at least one state")
    if isinstance(target_rank, bool) or not isinstance(target_rank, Integral) or target_rank < 0:
        raise ValueError("target_rank must be a nonnegative integer")
    target_rank = int(target_rank)
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    normalized = _weights(weights, len(states))
    if any(not isinstance(state, Mapping) for state in states):
        raise ValueError("states must be mappings")
    layers = set(states[0])
    if any(set(state) != layers for state in states):
        raise ValueError("all states must have identical layer sets")

    merged, layer_diagnostics = {}, {}
    total_energy = 0.0
    tail_energy = 0.0
    for layer in states[0]:
        (out_features, in_features), device, dtype = _validate_layer(states, layer)
        left_parts, right_parts = [], []
        ranks = []
        for state, weight in zip(states, normalized):
            a, b = state[layer]["A"], state[layer]["B"]
            rank = a.shape[0]
            ranks.append(rank)
            if rank == 0 or weight == 0:
                continue
            scale = math.sqrt(weight) * math.sqrt(alpha) / math.sqrt(rank)
            left_parts.append(b.to(dtype) * scale)
            right_parts.append(a.to(dtype).T * scale)

        width = sum(part.shape[1] for part in left_parts)
        q_left_width, q_right_width = min(out_features, width), min(in_features, width)
        retained = min(target_rank, q_left_width, q_right_width)
        output_a = torch.zeros(target_rank, in_features, dtype=dtype, device=device)
        output_b = torch.zeros(out_features, target_rank, dtype=dtype, device=device)

        if width:
            left, right = torch.cat(left_parts, dim=1), torch.cat(right_parts, dim=1)
            del left_parts, right_parts
            if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
                raise ValueError(f"layer {layer}: weighted factors overflowed")
            q_left, r_left = torch.linalg.qr(left, mode="reduced")
            q_right, r_right = torch.linalg.qr(right, mode="reduced")
            core = r_left @ r_right.T
            if not bool(torch.isfinite(core).all()):
                raise ValueError(f"layer {layer}: compact product overflowed")
            u, singular, vh = torch.linalg.svd(core, full_matrices=False)
            squared = singular.double().square()
            energy = float(squared.sum())
            discarded = float(squared[retained:].sum())
            if retained:
                # Remove the output alpha/r scale symmetrically from A and B.
                root = torch.sqrt(singular[:retained]) * (math.sqrt(target_rank) / math.sqrt(alpha))
                output_b[:, :retained] = (q_left @ u[:, :retained]) * root.unsqueeze(0)
                output_a[:retained, :] = root.unsqueeze(1) * (vh[:retained, :] @ q_right.T)
            if not bool(torch.isfinite(output_a).all()) or not bool(torch.isfinite(output_b).all()):
                raise ValueError(f"layer {layer}: output factorization overflowed")
            # Do not keep the preceding layer's QR/SVD tensors alive while
            # starting the next one; the output factors are the only carryover.
            del left, right, q_left, q_right, r_left, r_right, core, u, singular, vh, squared
            if retained:
                del root
        else:
            energy, discarded = 0.0, 0.0

        itemsize = torch.empty((), dtype=dtype).element_size()
        compact_dim = min(q_left_width, q_right_width)
        # Named algorithm intermediates, with concat source/destination copies.
        estimate_elements = (
            2 * (out_features + in_features) * width
            + out_features * q_left_width + in_features * q_right_width
            + (q_left_width + q_right_width) * width
            + q_left_width * q_right_width
            + (q_left_width + q_right_width + 1) * compact_dim
            + (out_features + in_features + 1) * retained
        )
        layer_diagnostics[layer] = {
            "input_ranks": ranks,
            "active_concatenated_rank": width,
            "target_rank": target_rank,
            "core_shape": [q_left_width, q_right_width],
            "core_reaches_dense_shape": q_left_width == out_features and q_right_width == in_features,
            "effective_update_energy": energy,
            "discarded_energy": discarded,
            "relative_discarded_energy": discarded / energy if energy else 0.0,
            "structural_workspace_estimate_bytes": estimate_elements * itemsize,
            "output_factor_bytes": (out_features + in_features) * target_rank * itemsize,
            "dense_delta_bytes_avoided": out_features * in_features * itemsize,
            "compute_dtype": str(dtype),
        }
        merged[layer] = {"A": output_a, "B": output_b}
        total_energy += energy
        tail_energy += discarded

    diagnostics = {
        "method": "weighted_factor_concatenation_thin_qr_core_svd",
        "normalized_weights": normalized,
        "layers": layer_diagnostics,
        "effective_update_energy": total_energy,
        "discarded_energy": tail_energy,
        "relative_discarded_energy": tail_energy / total_energy if total_energy else 0.0,
        "max_layer_structural_workspace_estimate_bytes": max(
            (d["structural_workspace_estimate_bytes"] for d in layer_diagnostics.values()), default=0
        ),
        "output_factor_bytes": sum(d["output_factor_bytes"] for d in layer_diagnostics.values()),
        "workspace_scope": (
            "Structural estimate of named merge tensors, not a measured peak; excludes input states, "
            "model/optimizer storage, allocator caches and QR/SVD backend workspace. Layers are sequential."
        ),
        "dense_effective_updates_materialized": False,
    }
    return merged, diagnostics
