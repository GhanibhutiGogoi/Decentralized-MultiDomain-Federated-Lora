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

    Compression : SVD truncation to target_rank principal components,
                  S[:r] * Vh[:r]. The singular values must be reapplied --
                  Vh alone has orthonormal rows, so dropping S makes the
                  output norm exactly sqrt(target_rank) regardless of the
                  input's magnitude, destroying all scale information.
    Expansion   : zero-padding along rank_dim.

    Fix: SVD of [m×n] with m < n yields Vh with only m rows. If those
    rows are fewer than target_rank (e.g. out_f=10 < FIXED_RANK=32),
    we zero-pad up to target_rank so the shape is always correct.

    Prefer `load_global_state`'s paired A/B path where possible: projecting
    the two factors independently is not the best rank-r approximation of
    the update B @ A.
    """
    cur_rank = t.shape[rank_dim]
    if cur_rank == target_rank:
        return t.clone()

    if cur_rank > target_rank:
        # Normalise so rank is always on dim-0 before SVD
        mat = t.float() if rank_dim == 0 else t.float().t()   # [cur_rank, d]
        _, S, Vh = torch.linalg.svd(mat, full_matrices=False)  # [min(cur,d), d]
        principal = S[:, None] * Vh                            # keep the scale
        actual_rows = principal.shape[0]
        if actual_rows >= target_rank:
            compressed = principal[:target_rank, :]            # [target_rank, d]
        else:
               
            pad = torch.zeros(
                target_rank - actual_rows, principal.shape[1],
                dtype=principal.dtype, device=principal.device)
            compressed = torch.cat([principal, pad], dim=0)     # [target_rank, d]
        result = compressed if rank_dim == 0 else compressed.t()
        return result.to(t.dtype)

    # cur_rank < target_rank → zero-pad along rank_dim
    pad_shape          = list(t.shape)
    pad_shape[rank_dim] = target_rank - cur_rank
    pad = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=rank_dim)


def load_global_state(model, global_state):
    """
    Load global_state into model, re-ranking LoRA matrices to the model's rank
    when sizes differ.

    Paired A/B keys are handled in *update space*: the global update
    Delta W = B_g A_g is reconstructed and refactorised at the local rank with
    the same truncated SVD the server uses (`_factorize_delta`). Projecting A
    and B independently is not the best rank-r approximation of B A, and A and
    B do not even live in comparable bases once ranks differ.

    Unpaired LoRA keys fall back to the elementwise projection:
      A keys: rank on dim-0
      B keys: rank on dim-1
    """
    # Imported here (not at module scope) because Federated.fedavg_aggregation
    # imports this module -- a top-level import would be circular.
    from Federated.fedavg_aggregation import _factorize_delta, _lora_pairs

    local = model.state_dict()
    handled = set()

    for a_key, b_key in _lora_pairs(local):
        if a_key not in global_state or b_key not in global_state:
            continue
        g_a, g_b = global_state[a_key], global_state[b_key]
        if g_a.dim() != 2 or g_b.dim() != 2 or g_b.shape[1] != g_a.shape[0]:
            continue

        l_a, l_b = local[a_key], local[b_key]
        if g_a.shape == l_a.shape and g_b.shape == l_b.shape:
            local[a_key] = g_a.clone()
            local[b_key] = g_b.clone()
            handled.update([a_key, b_key])
            continue

        if (g_b.shape[0], g_a.shape[1]) != (l_b.shape[0], l_a.shape[1]):
            # Not a rank mismatch but a genuine architecture mismatch, which no
            # re-ranking can reconcile. Falling through to the elementwise path
            # would hand load_state_dict a wrongly shaped tensor and fail with a
            # cryptic size-mismatch error, so say what is actually wrong.
            raise ValueError(
                f"layer geometry mismatch for {a_key!r}/{b_key!r}: global update is "
                f"{g_b.shape[0]}x{g_a.shape[1]} but the local model expects "
                f"{l_b.shape[0]}x{l_a.shape[1]}"
            )

        device = l_a.device
        delta = g_b.to(device).float() @ g_a.to(device).float()
        new_a, new_b = _factorize_delta(delta, l_a.shape[0], l_a.dtype)
        local[a_key] = new_a.to(device=device, dtype=l_a.dtype)
        local[b_key] = new_b.to(device=device, dtype=l_b.dtype)
        handled.update([a_key, b_key])

    for k in local:
        if k in handled or k not in global_state:
            continue
        g = global_state[k]
        if g.shape == local[k].shape:
            local[k] = g.clone()
        elif is_lora_key(k) and g.dim() == 2:
            rank_dim    = 1 if is_lora_B_key(k) else 0
            target_rank = local[k].shape[rank_dim]
            local[k]    = project_tensor_to_rank(g, target_rank, rank_dim)
    model.load_state_dict(local)
