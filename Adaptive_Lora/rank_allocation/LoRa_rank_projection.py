# LoRa rank projection utilities with some explanations provided.

import torch
from config import LORA_SUFFIXES, LORA_A_SUFFIXES, LORA_B_SUFFIXES


def is_lora_key(k):
    """True if parameter key belongs to a LoRA matrix (A or B)."""
    return any(k.endswith(s) for s in LORA_SUFFIXES)


def is_lora_B_key(k):
    """True if parameter key is a LoRA B matrix (shape [out_f, r])."""
    return any(k.endswith(s) for s in LORA_B_SUFFIXES)


def project_tensor_to_rank(t, target_rank, rank_dim=0):
    """
    Project a 2-D LoRA matrix to target_rank along rank_dim.
      rank_dim=0  →  A matrices, shape [r, in_f]
      rank_dim=1  →  B matrices, shape [out_f, r]

    Compression : SVD truncation to target_rank principal components.
    Expansion   : zero-padding along rank_dim.

    Fix: SVD of [m×n] with m < n yields Vh with only m rows. If those
    rows are fewer than target_rank (e.g. out_f=10 < FIXED_RANK=32),
    we zero-pad up to target_rank so the shape is always correct.
    """
    cur_rank = t.shape[rank_dim]
    if cur_rank == target_rank:
        return t.clone()

    if cur_rank > target_rank:
        # Normalise so rank is always on dim-0 before SVD
        mat = t.float() if rank_dim == 0 else t.float().t()   # [cur_rank, d]
        _, _, Vh = torch.linalg.svd(mat, full_matrices=False)  # [min(cur,d), d]
        actual_rows = Vh.shape[0]
        if actual_rows >= target_rank:
            compressed = Vh[:target_rank, :]                   # [target_rank, d]
        else:
               
            pad = torch.zeros(
                target_rank - actual_rows, Vh.shape[1],
                dtype=Vh.dtype, device=Vh.device)
            compressed = torch.cat([Vh, pad], dim=0)           # [target_rank, d]
        result = compressed if rank_dim == 0 else compressed.t()
        return result.to(t.dtype)

    # cur_rank < target_rank → zero-pad along rank_dim
    pad_shape          = list(t.shape)
    pad_shape[rank_dim] = target_rank - cur_rank
    pad = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=rank_dim)


def load_global_state(model, global_state):
    """
    Load global_state into model, projecting LoRA matrices to the
    model's rank when sizes differ.
      A keys: rank on dim-0
      B keys: rank on dim-1
    """
    local = model.state_dict()
    for k in local:
        if k not in global_state:
            continue
        g = global_state[k]
        if g.shape == local[k].shape:
            local[k] = g.clone()
        elif is_lora_key(k) and g.dim() == 2:
            rank_dim    = 1 if is_lora_B_key(k) else 0
            target_rank = local[k].shape[rank_dim]
            local[k]    = project_tensor_to_rank(g, target_rank, rank_dim)
    model.load_state_dict(local)
