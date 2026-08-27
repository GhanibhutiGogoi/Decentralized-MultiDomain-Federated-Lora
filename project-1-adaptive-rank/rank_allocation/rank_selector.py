"""Capability-aware adaptive LoRA rank selection."""

import math

import numpy as np
import torch

from config import ALL_CANDIDATE_RANKS, BATCH_TO_MAX_RANK
from Federated.client import set_lora_only_trainable

# Weight on the capability prior in the rank floor. gamma < 1 keeps the
# capability term a soft prior so the measured demand term can bind; gamma = 1
# reproduces the pre-fix behaviour, where the floor equalled the hardware
# ceiling at the top tier and the stable-rank measurement was discarded.
#
# 0.5 is the midpoint of the usable interval (0, 1) and is deliberately not
# tuned: it reserves half of each client's hardware budget as a floor and
# leaves the other half for the gradient measurement to claim. It is a
# published constant of the allocation rule, so it is recorded alongside the
# equation in the experiment artifact rather than left implicit. Note the
# consequence at the middle tier of the shipped BATCH_TO_MAX_RANK: the floor
# gamma * 0.5 * 8 = 2 coincides with the smallest candidate rank, so there the
# prior is inert by construction and demand alone decides.
GAMMA = 0.5


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


def rank_equation(stable_rank, batch_size, gamma=GAMMA):
    r"""
    Closed-form adaptive rank rule.

        s(G) = ||G||_F^2 / ||G||_2^2
        c_i  = (index(batch_i) / (num_capabilities - 1))
        r_i  = round_to_candidate(
                 max(2, gamma * c_i * R_i^max, min(s(G), R_i^max))
               )

    The stable rank estimates update complexity from the gradient geometry.
    The capability term is a *soft prior* (weight gamma < 1) that keeps stronger
    clients from being under-allocated without pinning them to their ceiling;
    the ceiling prevents weaker clients from exceeding their budget.

    With gamma = 1 the floor equals R_i^max whenever c_i = 1, and since the
    demand term is capped at R_i^max the outer max always returns the floor --
    the measured stable rank is computed and then discarded. gamma < 1 leaves
    room between the floor and the ceiling for s(G) to bind.
    """
    max_rank = BATCH_TO_MAX_RANK.get(batch_size, min(ALL_CANDIDATE_RANKS))
    candidates = [r for r in ALL_CANDIDATE_RANKS if r <= max_rank]
    if not candidates:
        return min(ALL_CANDIDATE_RANKS)

    # A non-finite measurement means the gradient probe diverged. Fall back to
    # the capability floor deliberately rather than letting NaN propagate
    # silently through max/min, where it would collapse to the floor anyway but
    # by accident.
    demand = float(stable_rank)
    if not math.isfinite(demand):
        demand = 0.0

    floor = max(candidates[0], gamma * capability_fraction(batch_size) * max_rank)
    raw_rank = max(floor, min(demand, float(max_rank)))

    chosen = _nearest_candidate(raw_rank, candidates)
    # _nearest_candidate breaks ties downward, which can land below the floor
    # when the floor sits between two candidates. Snap back up so the floor is
    # a genuine lower bound for any gamma, not just the shipped one.
    if chosen < floor:
        above = [r for r in candidates if r >= floor]
        if above:
            chosen = min(above)
    return chosen


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
