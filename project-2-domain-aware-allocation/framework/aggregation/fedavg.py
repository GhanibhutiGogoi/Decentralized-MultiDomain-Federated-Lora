"""Server-side aggregation for heterogeneous LoRA ranks.

This is the Project 1 quality-weighted aggregation implementation migrated
into the reusable framework namespace without changing its behavior.
"""

import torch

from framework.aggregation.projection import (
    LORA_A_SUFFIXES,
    LORA_B_SUFFIXES,
    is_lora_key,
)


def _normalised_client_weights(samples, quality_scores):
    raw = [float(s) * float(q) for s, q in zip(samples, quality_scores)]
    total = sum(raw)
    if total <= 0:
        return [1.0 / len(raw)] * len(raw)
    return [r / total for r in raw]


def _matching_b_key(a_key):
    for a_suffix, b_suffix in zip(LORA_A_SUFFIXES, LORA_B_SUFFIXES):
        if a_key.endswith(a_suffix):
            return a_key[: -len(a_suffix)] + b_suffix
    return None


def _lora_pairs(ref_sd):
    pairs = []
    for a_key in ref_sd:
        b_key = _matching_b_key(a_key)
        if b_key and b_key in ref_sd:
            pairs.append((a_key, b_key))
    return pairs


def _factorize_delta(delta, target_rank, dtype):
    """Return A, B such that B @ A approximates delta at target_rank."""
    out_f, in_f = delta.shape
    rank = min(target_rank, out_f, in_f)
    u, s, vh = torch.linalg.svd(delta.float(), full_matrices=False)

    if rank > 0:
        root_s = torch.sqrt(torch.clamp(s[:rank], min=0.0))
        b = u[:, :rank] * root_s.unsqueeze(0)
        a = root_s.unsqueeze(1) * vh[:rank, :]
    else:
        b = torch.zeros(out_f, 0, device=delta.device)
        a = torch.zeros(0, in_f, device=delta.device)

    if target_rank > rank:
        b_pad = torch.zeros(out_f, target_rank - rank, device=delta.device)
        a_pad = torch.zeros(target_rank - rank, in_f, device=delta.device)
        b = torch.cat([b, b_pad], dim=1)
        a = torch.cat([a, a_pad], dim=0)

    return a.to(dtype), b.to(dtype)


def fedavg_quality_weighted(weights, samples, quality_scores, target_rank, ref_sd, device):
    """
    FedAvg weighted by samples times quality, with heterogeneous LoRA handled
    in update space.
    """
    norm_w = _normalised_client_weights(samples, quality_scores)
    agg = {}
    handled = set()

    for a_key, b_key in _lora_pairs(ref_sd):
        deltas = []
        for state, client_weight in zip(weights, norm_w):
            if a_key not in state or b_key not in state:
                continue
            a = state[a_key].to(device)
            b = state[b_key].to(device)
            if a.dim() != 2 or b.dim() != 2 or b.shape[1] != a.shape[0]:
                continue
            deltas.append((b.float() @ a.float(), client_weight))

        if deltas:
            delta = sum(d * w for d, w in deltas)
            a, b = _factorize_delta(delta, target_rank, ref_sd[a_key].dtype)
            agg[a_key] = a
            agg[b_key] = b
        else:
            agg[a_key] = ref_sd[a_key]
            agg[b_key] = ref_sd[b_key]
        handled.update([a_key, b_key])

    for key in ref_sd:
        if key in handled:
            continue
        if is_lora_key(key):
            agg[key] = ref_sd[key]
            continue
        contribs = [
            (state[key].to(device), client_weight)
            for state, client_weight in zip(weights, norm_w)
            if key in state and state[key].shape == ref_sd[key].shape
        ]
        agg[key] = sum(t * w for t, w in contribs) if contribs else ref_sd[key]

    return agg

