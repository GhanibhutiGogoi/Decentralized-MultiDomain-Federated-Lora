# FedAvg aggregation

import torch
from rank_allocation.LoRa_rank_projection import is_lora_key, is_lora_B_key, project_tensor_to_rank


def fedavg_quality_weighted(weights, samples, quality_scores,
                             target_rank, ref_sd, device):
    """
    FedAvg weighted by samples × quality_score.

    A client with high loss (struggling due to rank too large for its
    capability) contributes less to the global model. LoRA matrices are
    projected to target_rank before averaging so all clients contribute
    to a common rank regardless of their local rank choice.

    Args:
        weights       : list of client state_dicts
        samples       : list of sample counts per client
        quality_scores: list of quality scores per client
        target_rank   : rank of the global model (FIXED_RANK)
        ref_sd        : reference state_dict (shape template)
        device        : torch device

    Returns:
        aggregated state_dict at target_rank
    """
    raw    = [s * q for s, q in zip(samples, quality_scores)]
    total  = sum(raw)
    norm_w = [r / total for r in raw]

    agg = {}
    for k in ref_sd:
        if is_lora_key(k):
            rank_dim  = 1 if is_lora_B_key(k) else 0
            projected = []
            for w, nw in zip(weights, norm_w):
                if k not in w or w[k].dim() != 2:
                    continue
                t = project_tensor_to_rank(
                    w[k].to(device), target_rank, rank_dim)
                projected.append((t, nw))
            agg[k] = (sum(t * nw for t, nw in projected)
                      if projected else ref_sd[k])
        else:
            contribs = [
                (w[k], nw)
                for w, nw in zip(weights, norm_w)
                if k in w and w[k].shape == ref_sd[k].shape
            ]
            agg[k] = (sum(t * nw for t, nw in contribs)
                      if contribs else ref_sd[k])

    return agg
