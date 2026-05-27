# capability-aware adaptive rank selector

import numpy as np
import torch
import torch.optim as optim
from config import BATCH_TO_MAX_RANK, ALL_CANDIDATE_RANKS


def estimate_optimal_rank(model, loader, loss_fn, batch_size, num_batches=3):
    """
    Gradient-signal rank probe with batch-size-based capability floor.

    Candidate ranks are capped by BATCH_TO_MAX_RANK so weak clients
    (small batch) never choose a rank beyond their capability.

    capability_bias acts as a hard floor on the candidate index:
      C0 (bs=16 ) → bias=0.0 → floor at idx 0   → min r=2
      C1 (bs=64 ) → bias=0.5 → floor at idx 50% → min r=mid candidate
      C2 (bs=256) → bias=1.0 → floor at idx 100% → always max candidate

    The gradient stable-rank signal can only push the chosen index
    higher than the floor, never lower.
    """
    max_rank   = BATCH_TO_MAX_RANK.get(batch_size, 4)
    candidates = [r for r in ALL_CANDIDATE_RANKS if r <= max_rank]

    sorted_bs       = sorted(BATCH_TO_MAX_RANK.keys())
    bs_idx          = sorted_bs.index(batch_size)
    capability_bias = bs_idx / (len(sorted_bs) - 1)

    model.train()
    opt          = optim.Adam(model.parameters(), lr=0.001)
    stable_ranks = []

    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= num_batches:
            break
        x, y = x.to(next(model.parameters()).device), \
               y.to(next(model.parameters()).device)
        opt.zero_grad()
        loss_fn(model(x), y).backward()

        for _, param in model.named_parameters():
            if param.grad is not None and param.dim() == 2:
                G           = param.grad.float()
                frob_sq     = (G ** 2).sum().item()
                spectral_sq = torch.linalg.matrix_norm(G, ord=2).item() ** 2
                if spectral_sq > 1e-12:
                    stable_ranks.append(frob_sq / spectral_sq)

    if not stable_ranks:
        return candidates[-1]

    median_sr   = float(np.median(stable_ranks))
    signal_frac = min(median_sr / max(ALL_CANDIDATE_RANKS), 1.0)
    combined    = max(signal_frac, capability_bias)
    idx         = int(round(combined * (len(candidates) - 1)))
    return candidates[min(idx, len(candidates) - 1)]
