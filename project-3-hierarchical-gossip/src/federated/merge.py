"""Delta-W merge kernels for LoRA aggregation under heterogeneous ranks.

Gossip in `gossip.py` averages the A and B factors separately. That is
ill-posed: for any invertible Q the pairs (B, A) and (B Q^T, Q A) encode the
identical update, so factor averaging depends on a gauge the clients never
agreed on, and it cannot run at all once two clients hold different ranks.

Everything here operates on the *effective* update actually applied by the
forward pass,

    Delta W = (alpha / r) * B @ A,

which is gauge invariant and has shape (out, in) whatever the rank. Merging
the unscaled B @ A is wrong the moment ranks differ, because alpha / r then
differs across clients by the ratio of their ranks.

Refactorisation is the truncated SVD with the sqrt(S) split, carrying the
1 / sqrt(alpha / target_rank) correction so that (alpha / target_rank) * B @ A
reproduces the merged update. Provenance: ported from
`project-2-domain-aware-allocation/src/models/lora_resnet.py::decompose_delta_w`
(the scale correction) and `project-1-adaptive-rank/Federated/
fedavg_aggregation.py::_factorize_delta` (the padding behaviour). Copied rather
than imported: projects in this repo are self-contained.
"""

import math

import torch


def _check_alpha(alpha):
    value = float(alpha)
    # NaN is already excluded by `not (nan > 0)`, but say so explicitly rather
    # than relying on that: alpha = inf otherwise passes the positivity test and
    # then makes scaling = sqrt(inf / r) = inf, so b / scaling silently returns
    # an all-zero adapter from factorize_delta and non-finite values from
    # lora_to_delta. A client that quietly learns nothing is worse than a crash.
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"alpha must be finite and > 0, got {alpha}")


def _compute_dtype(dtype):
    """Work in float32 for headroom, but never *below* the caller's precision."""
    return torch.float64 if dtype == torch.float64 else torch.float32


def _layer_delta(params, alpha):
    """(alpha / r) * B @ A for one layer, with the rank-0 guard."""
    a, b = params['A'], params['B']
    rank = a.shape[0]
    work = _compute_dtype(a.dtype)
    if rank == 0:
        return torch.zeros(b.shape[0], a.shape[1], dtype=work, device=b.device)
    return (float(alpha) / rank) * (b.to(work) @ a.to(work))


def lora_to_delta(lora_state, alpha):
    """Per-layer effective update Delta W = (alpha / r) * B @ A.

    Args:
        lora_state: {layer_name: {'A': [r, in], 'B': [out, r]}}
        alpha: LoRA alpha used by the forward pass.

    Returns:
        {layer_name: tensor [out, in]}, in float32 working precision -- promoted
        to float64 when the caller's factors are float64, so a double-precision
        model is not silently downgraded. float16 and bfloat16 inputs still
        compute in float32.
    """
    _check_alpha(alpha)
    return {layer: _layer_delta(params, alpha) for layer, params in lora_state.items()}


def factorize_delta(delta, target_rank, alpha, dtype=None):
    """Factor a dense Delta W back into LoRA A/B at target_rank.

    Returns {'A': [target_rank, in], 'B': [out, target_rank]} such that
    (alpha / target_rank) * B @ A is the best rank-target_rank approximation of
    `delta` (exactly `delta` when target_rank >= rank(delta)).
    """
    if target_rank < 1:
        raise ValueError(f"target_rank must be >= 1, got {target_rank}")
    _check_alpha(alpha)
    if dtype is None:
        dtype = delta.dtype

    out_f, in_f = delta.shape
    u, s, vh = torch.linalg.svd(delta.to(_compute_dtype(dtype)), full_matrices=False)
    keep = min(target_rank, s.shape[0])

    root_s = torch.sqrt(torch.clamp(s[:keep], min=0.0))
    b = u[:, :keep] * root_s.unsqueeze(0)      # [out, keep]
    a = root_s.unsqueeze(1) * vh[:keep, :]     # [keep, in]

    # The forward pass multiplies by alpha / target_rank; divide it out here so
    # the reconstruction is the merged delta itself and not a rescaled copy.
    scaling = math.sqrt(float(alpha) / target_rank)
    b = b / scaling
    a = a / scaling

    if keep < target_rank:
        pad = target_rank - keep
        b = torch.cat(
            [b, torch.zeros(out_f, pad, dtype=b.dtype, device=b.device)], dim=1
        )
        a = torch.cat(
            [a, torch.zeros(pad, in_f, dtype=a.dtype, device=a.device)], dim=0
        )

    return {'A': a.to(dtype), 'B': b.to(dtype)}


def _normalised(weights, n_states):
    if len(weights) != n_states:
        raise ValueError(
            f"expected {n_states} weights, one per state, got {len(weights)}"
        )
    # Validate each weight, not just the total. [-1, 2] sums to 1 and would be
    # accepted by a total-only check, but it extrapolates away from both clients
    # instead of averaging them -- on two real states it produced a merged update
    # of norm 301 from inputs of norm 141 and 120, i.e. outside their convex
    # hull. Non-finite weights are worse still: they survive the sum and only
    # surface much later as a bare torch _LinAlgError from inside the SVD,
    # which names ill-conditioning rather than the bad input that caused it.
    #
    # Negative weights are rejected rather than supported. These are aggregation
    # weights, so a negative one means "move away from this client", which no
    # weighted average should do. Some accelerated-gossip schemes do use signed
    # mixing weights; if project 3 ever wants that it should be a separate,
    # explicitly named entry point, not a silent reinterpretation of this one.
    for k, w in enumerate(weights):
        value = float(w)
        if not math.isfinite(value):
            raise ValueError(f"weight {k} must be finite, got {w}")
        if value < 0.0:
            raise ValueError(
                f"weight {k} must be >= 0, got {w}; negative weights extrapolate "
                f"away from a client rather than averaging toward it"
            )

    total = float(sum(float(w) for w in weights))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"weights must sum to a finite positive total, got {total}")
    return [float(w) / total for w in weights]


def merge_states(states, weights, target_rank, alpha):
    """Weighted merge of LoRA states in delta-W space, refactorised to target_rank.

    States may carry different ranks. Weights are normalised to sum to 1.

    Returns a lora_state {layer: {'A', 'B'}} at target_rank.
    """
    if not states:
        raise ValueError("merge_states needs at least one state")
    # Validate alpha at the public entry point, before any weight or layer work.
    # Downstream factorize_delta also checks it, but relying on that alone leaves
    # two gaps: an invalid alpha is used to build intermediate deltas in
    # _layer_delta before anything rejects it, and for a state with no layers the
    # factorization path is never reached at all, so merge_states([{}, {}], w, r,
    # float("inf")) returned {} instead of raising.
    _check_alpha(alpha)
    norm_w = _normalised(weights, len(states))

    layers = set(states[0])
    for k, state in enumerate(states[1:], start=1):
        if set(state) != layers:
            missing, extra = layers - set(state), set(state) - layers
            raise ValueError(
                f"state {k} has a different layer set: "
                f"missing {sorted(missing)}, unexpected {sorted(extra)}"
            )

    if not layers:
        return {}

    merged = {}
    for layer in states[0]:
        # Per-layer, so a mixed-precision model keeps each layer's own dtype
        # rather than having one arbitrary layer's dtype applied to all of them.
        out_dtype = states[0][layer]['A'].dtype
        total = None
        # One layer at a time via the shared helper: this keeps the rank-0 guard
        # and the scaling rule in a single place, without materialising every
        # state's every layer as a dense delta up front (which put peak memory
        # at n_states * n_layers dense matrices instead of two).
        for state, weight in zip(states, norm_w):
            contribution = weight * _layer_delta(state[layer], alpha)
            total = contribution if total is None else total + contribution
        merged[layer] = factorize_delta(total, target_rank, alpha, dtype=out_dtype)
    return merged
