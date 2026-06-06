"""Capability-aware adaptive LoRA rank selection."""

import numpy as np
import torch

from config import ALL_CANDIDATE_RANKS, BATCH_TO_MAX_RANK
from Federated.client import set_lora_only_trainable


def _nearest_candidate(rank, candidates):
    return min(candidates, key=lambda r: (abs(r - rank), r))


def capability_fraction(batch_size):
    """Map client batch-size capability to [0, 1]."""
    batch_sizes = sorted(BATCH_TO_MAX_RANK)
    if batch_size not in BATCH_TO_MAX_RANK:
        return 0.0
    if len(batch_sizes) == 1:
        return 1.0
    return batch_sizes.index(batch_size) / (len(batch_sizes) - 1)


def rank_equation(stable_rank, batch_size):
    r"""
    Closed-form adaptive rank rule.

        s(G) = ||G||_F^2 / ||G||_2^2
        c_i  = (index(batch_i) / (num_capabilities - 1))
        r_i  = round_to_candidate(
                 max(2, c_i * R_i^max, min(s(G), R_i^max))
               )

    The stable rank estimates update complexity from the gradient geometry.
    The capability term prevents stronger clients from being under-allocated,
    while the ceiling prevents weaker clients from exceeding their budget.
    """
    max_rank = BATCH_TO_MAX_RANK.get(batch_size, min(ALL_CANDIDATE_RANKS))
    candidates = [r for r in ALL_CANDIDATE_RANKS if r <= max_rank]
    if not candidates:
        return min(ALL_CANDIDATE_RANKS)

    floor = max(candidates[0], capability_fraction(batch_size) * max_rank)
    raw_rank = max(floor, min(float(stable_rank), float(max_rank)))
    return _nearest_candidate(raw_rank, candidates)


def estimate_gradient_stable_rank(model, loader, loss_fn, num_batches=3):
    """Estimate median gradient stable rank from trainable 2-D parameters."""
    model.train()
    set_lora_only_trainable(model)
    stable_ranks = []
    device = next(model.parameters()).device

    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= num_batches:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()

        for param in model.parameters():
            if param.requires_grad and param.grad is not None and param.grad.dim() == 2:
                grad = param.grad.float()
                frob_sq = torch.sum(grad * grad).item()
                spectral_sq = torch.linalg.matrix_norm(grad, ord=2).item() ** 2
                if spectral_sq > 1e-12:
                    stable_ranks.append(frob_sq / spectral_sq)

    return float(np.median(stable_ranks)) if stable_ranks else 1.0


def estimate_optimal_rank(model, loader, loss_fn, batch_size, num_batches=3):
    """Select adaptive rank using the closed-form rank equation."""
    stable_rank = estimate_gradient_stable_rank(model, loader, loss_fn, num_batches)
    return rank_equation(stable_rank, batch_size)
