"""LoRA rank projection utilities used during aggregation.

This is the Project 1 implementation migrated into the reusable framework
namespace without changing the projection behavior.
"""

import torch


LORA_A_SUFFIXES = (".A", ".lora_q_A", ".lora_k_A", ".lora_v_A")
LORA_B_SUFFIXES = (".B", ".lora_q_B", ".lora_k_B", ".lora_v_B")
LORA_SUFFIXES = LORA_A_SUFFIXES + LORA_B_SUFFIXES


def is_lora_key(k):
    """True if parameter key belongs to a LoRA matrix (A or B)."""
    return any(k.endswith(s) for s in LORA_SUFFIXES)


def is_lora_B_key(k):
    """True if parameter key is a LoRA B matrix (shape [out_f, r])."""
    return any(k.endswith(s) for s in LORA_B_SUFFIXES)


def project_tensor_to_rank(t, target_rank, rank_dim=0):
    """
    Project a 2-D LoRA matrix to target_rank along rank_dim.
      rank_dim=0 -> A matrices, shape [r, in_f]
      rank_dim=1 -> B matrices, shape [out_f, r]
    """
    cur_rank = t.shape[rank_dim]
    if cur_rank == target_rank:
        return t.clone()

    if cur_rank > target_rank:
        mat = t.float() if rank_dim == 0 else t.float().t()
        _, _, Vh = torch.linalg.svd(mat, full_matrices=False)
        actual_rows = Vh.shape[0]
        if actual_rows >= target_rank:
            compressed = Vh[:target_rank, :]
        else:
            pad = torch.zeros(
                target_rank - actual_rows,
                Vh.shape[1],
                dtype=Vh.dtype,
                device=Vh.device,
            )
            compressed = torch.cat([Vh, pad], dim=0)
        result = compressed if rank_dim == 0 else compressed.t()
        return result.to(t.dtype)

    pad_shape = list(t.shape)
    pad_shape[rank_dim] = target_rank - cur_rank
    pad = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=rank_dim)


def load_global_state(model, global_state):
    """Load global state, projecting LoRA matrices to the model rank."""
    local = model.state_dict()
    for k in local:
        if k not in global_state:
            continue
        g = global_state[k]
        if g.shape == local[k].shape:
            local[k] = g.clone()
        elif is_lora_key(k) and g.dim() == 2:
            rank_dim = 1 if is_lora_B_key(k) else 0
            target_rank = local[k].shape[rank_dim]
            local[k] = project_tensor_to_rank(g, target_rank, rank_dim)
    model.load_state_dict(local)

