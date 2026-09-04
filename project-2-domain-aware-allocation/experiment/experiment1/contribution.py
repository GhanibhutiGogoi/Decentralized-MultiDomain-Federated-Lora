"""Leave-one-client-out contribution and update-vector utilities."""

from __future__ import annotations

import copy

import numpy as np
import torch


def _matching_b_key(a_key: str, a_suffixes, b_suffixes):
    for a_suffix, b_suffix in zip(a_suffixes, b_suffixes):
        if a_key.endswith(a_suffix):
            return a_key[: -len(a_suffix)] + b_suffix
    return None


def _lora_pairs(state, a_suffixes, b_suffixes):
    pairs = []
    for key in state:
        b_key = _matching_b_key(key, a_suffixes, b_suffixes)
        if b_key and b_key in state:
            pairs.append((key, b_key))
    return pairs


def flatten_update_vector(
    client_state,
    global_state,
    a_suffixes,
    b_suffixes,
    lora_suffixes,
) -> np.ndarray:
    """Flatten a client update in rank-independent LoRA delta-W space."""
    parts = []
    handled = set()

    for a_key, b_key in _lora_pairs(client_state, a_suffixes, b_suffixes):
        if a_key not in global_state or b_key not in global_state:
            continue
        client_a = client_state[a_key].detach().float().cpu()
        client_b = client_state[b_key].detach().float().cpu()
        global_a = global_state[a_key].detach().float().cpu()
        global_b = global_state[b_key].detach().float().cpu()
        if client_a.dim() != 2 or client_b.dim() != 2:
            continue
        if global_a.dim() != 2 or global_b.dim() != 2:
            continue
        client_delta = client_b @ client_a
        global_delta = global_b @ global_a
        if client_delta.shape == global_delta.shape:
            parts.append((client_delta - global_delta).reshape(-1).numpy())
        handled.update([a_key, b_key])

    for key, value in client_state.items():
        if key in handled or any(key.endswith(suffix) for suffix in lora_suffixes):
            continue
        if key not in global_state or value.shape != global_state[key].shape:
            continue
        delta = value.detach().float().cpu() - global_state[key].detach().float().cpu()
        if torch.count_nonzero(delta).item():
            parts.append(delta.reshape(-1).numpy())

    if not parts:
        return np.zeros(1, dtype=float)
    return np.concatenate(parts).astype(float)


def evaluate_leave_one_client_out(
    weights,
    samples,
    quality_scores,
    target_rank,
    ref_sd,
    model_fn,
    testloader,
    device,
    aggregate_fn,
    evaluate_fn,
    load_global_state_fn,
) -> tuple[dict, list[dict]]:
    """Evaluate full aggregation and one aggregation excluding each client."""
    full_state = aggregate_fn(weights, samples, quality_scores, target_rank, ref_sd, device)
    full_model = model_fn(target_rank).to(device)
    load_global_state_fn(full_model, full_state)
    full_accuracy = evaluate_fn(full_model, testloader, device)

    rows = []
    for client_id in range(len(weights)):
        keep = [idx for idx in range(len(weights)) if idx != client_id]
        if not keep:
            loo_accuracy = full_accuracy
        else:
            loo_state = aggregate_fn(
                [weights[idx] for idx in keep],
                [samples[idx] for idx in keep],
                [quality_scores[idx] for idx in keep],
                target_rank,
                copy.deepcopy(ref_sd),
                device,
            )
            loo_model = model_fn(target_rank).to(device)
            load_global_state_fn(loo_model, loo_state)
            loo_accuracy = evaluate_fn(loo_model, testloader, device)

        rows.append(
            {
                "client_id": client_id,
                "full_accuracy": float(full_accuracy),
                "loo_accuracy": float(loo_accuracy),
                "delta_accuracy": float(full_accuracy - loo_accuracy),
            }
        )

    return full_state, rows

