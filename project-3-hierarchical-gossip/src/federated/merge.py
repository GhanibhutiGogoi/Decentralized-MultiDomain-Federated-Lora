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


def lora_to_delta(lora_state, alpha):
    """Per-layer effective update Delta W = (alpha / r) * B @ A.

    Args:
        lora_state: {layer_name: {'A': [r, in], 'B': [out, r]}}
        alpha: LoRA alpha used by the forward pass.

    Returns:
        {layer_name: tensor [out, in]}
    """
    deltas = {}
    for layer, params in lora_state.items():
        a, b = params['A'], params['B']
        rank = a.shape[0]
        if rank == 0:
            deltas[layer] = torch.zeros(b.shape[0], a.shape[1], dtype=a.dtype)
            continue
        deltas[layer] = (float(alpha) / rank) * (b.float() @ a.float())
    return deltas


def factorize_delta(delta, target_rank, alpha):
    """Factor a dense Delta W back into LoRA A/B at target_rank.

    Returns {'A': [target_rank, in], 'B': [out, target_rank]} such that
    (alpha / target_rank) * B @ A is the best rank-target_rank approximation of
    `delta` (exactly `delta` when target_rank >= rank(delta)).
    """
    if target_rank < 1:
        raise ValueError(f"target_rank must be >= 1, got {target_rank}")

    out_f, in_f = delta.shape
    u, s, vh = torch.linalg.svd(delta.float(), full_matrices=False)
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
        b = torch.cat([b, torch.zeros(out_f, pad, dtype=b.dtype)], dim=1)
        a = torch.cat([a, torch.zeros(pad, in_f, dtype=a.dtype)], dim=0)

    return {'A': a.to(delta.dtype), 'B': b.to(delta.dtype)}


def _normalised(weights, n_states):
    if len(weights) != n_states:
        raise ValueError(
            f"expected {n_states} weights, one per state, got {len(weights)}"
        )
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError(f"weights must sum to a positive total, got {total}")
    return [float(w) / total for w in weights]


def merge_states(states, weights, target_rank, alpha):
    """Weighted merge of LoRA states in delta-W space, refactorised to target_rank.

    States may carry different ranks. Weights are normalised to sum to 1.

    Returns a lora_state {layer: {'A', 'B'}} at target_rank.
    """
    if not states:
        raise ValueError("merge_states needs at least one state")
    norm_w = _normalised(weights, len(states))

    merged = {}
    for layer in states[0]:
        total = None
        for state, weight in zip(states, norm_w):
            if layer not in state:
                raise ValueError(f"layer {layer!r} missing from one of the states")
            params = state[layer]
            rank = params['A'].shape[0]
            delta = (float(alpha) / rank) * (params['B'].float() @ params['A'].float())
            total = weight * delta if total is None else total + weight * delta
        merged[layer] = factorize_delta(total, target_rank, alpha)
    return merged
