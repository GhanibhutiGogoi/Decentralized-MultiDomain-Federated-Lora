"""LoRA rank projection utilities used during aggregation.

These helpers use the unscaled ``B @ A`` convention of the Project 1 models.
The alpha/r-scaled ResNet helpers live in ``framework.models.lora_resnet``.
"""

import numbers

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


def _validated_svd_input(t, *, label: str) -> torch.Tensor:
    if not torch.is_floating_point(t):
        raise ValueError(f"{label} must be a floating-point tensor")
    if t.numel() == 0:
        raise ValueError(f"{label} must be non-empty")
    if not torch.isfinite(t).all():
        raise ValueError(f"{label} must contain only finite values")
    return t if t.dtype == torch.float64 else t.float()


def project_tensor_to_rank(t, target_rank, rank_dim=0):
    """
    Project a 2-D LoRA matrix to target_rank along rank_dim.
      rank_dim=0 -> A matrices, shape [r, in_f]
      rank_dim=1 -> B matrices, shape [out_f, r]

    Retain singular values during compression so update magnitude is not
    discarded. When both factors are available, prefer ``load_global_state``:
    independently projecting A and B does not approximate their product.
    """
    if not isinstance(target_rank, numbers.Integral) or isinstance(target_rank, bool) or target_rank <= 0:
        raise ValueError("target_rank must be a positive integer")
    target_rank = int(target_rank)
    if t.dim() != 2:
        raise ValueError("LoRA tensors must be 2-D")
    if rank_dim not in (0, 1):
        raise ValueError("rank_dim must be 0 or 1")
    cur_rank = t.shape[rank_dim]
    if cur_rank <= 0:
        raise ValueError("LoRA rank dimension must be non-empty")
    _validated_svd_input(t, label="LoRA tensor")
    if cur_rank == target_rank:
        return t.clone()

    if cur_rank > target_rank:
        work = t if t.dtype == torch.float64 else t.float()
        mat = work if rank_dim == 0 else work.t()
        _, singular_values, Vh = torch.linalg.svd(mat, full_matrices=False)
        principal = singular_values[:, None] * Vh
        actual_rows = principal.shape[0]
        if actual_rows >= target_rank:
            compressed = principal[:target_rank, :]
        else:
            pad = torch.zeros(
                target_rank - actual_rows,
                principal.shape[1],
                dtype=principal.dtype,
                device=principal.device,
            )
            compressed = torch.cat([principal, pad], dim=0)
        result = compressed if rank_dim == 0 else compressed.t()
        return result.to(t.dtype)

    pad_shape = list(t.shape)
    pad_shape[rank_dim] = target_rank - cur_rank
    pad = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=rank_dim)


def load_global_state(model, global_state):
    """Load global state, projecting paired LoRA updates to the model rank.

    Reconstruct ``B @ A`` and take its best truncated-SVD approximation when
    paired factor ranks differ. Same-shape factors are copied verbatim, and
    unpaired legacy keys retain the single-factor projection fallback.
    """
    # fedavg imports the suffix helpers above, so defer this import to avoid
    # a circular module initialization.
    from framework.aggregation.fedavg import _factorize_delta, _lora_pairs

    local = model.state_dict()
    handled = set()

    for a_key, b_key in _lora_pairs(local):
        if a_key not in global_state or b_key not in global_state:
            continue
        g_a, g_b = global_state[a_key], global_state[b_key]
        if g_a.dim() != 2 or g_b.dim() != 2 or g_b.shape[1] != g_a.shape[0]:
            continue
        _validated_svd_input(g_a, label=f"global LoRA tensor {a_key!r}")
        _validated_svd_input(g_b, label=f"global LoRA tensor {b_key!r}")

        l_a, l_b = local[a_key], local[b_key]
        if l_b.shape[1] != l_a.shape[0]:
            raise ValueError(
                f"local LoRA pair {a_key!r}/{b_key!r} is inconsistent: B has "
                f"{l_b.shape[1]} columns but A has {l_a.shape[0]} rows"
            )
        if g_a.shape == l_a.shape and g_b.shape == l_b.shape:
            local[a_key] = g_a.clone()
            local[b_key] = g_b.clone()
            handled.update([a_key, b_key])
            continue

        if (g_b.shape[0], g_a.shape[1]) != (l_b.shape[0], l_a.shape[1]):
            raise ValueError(
                f"layer geometry mismatch for {a_key!r}/{b_key!r}: global update is "
                f"{g_b.shape[0]}x{g_a.shape[1]} but the local model expects "
                f"{l_b.shape[0]}x{l_a.shape[1]}"
            )

        device = l_a.device
        work_dtype = torch.float64 if torch.float64 in (g_a.dtype, g_b.dtype) else torch.float32
        delta = g_b.to(device=device, dtype=work_dtype) @ g_a.to(device=device, dtype=work_dtype)
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
            rank_dim = 1 if is_lora_B_key(k) else 0
            target_rank = local[k].shape[rank_dim]
            local[k] = project_tensor_to_rank(g, target_rank, rank_dim)
    model.load_state_dict(local)
