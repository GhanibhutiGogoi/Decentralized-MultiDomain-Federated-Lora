"""Tests for LoRA rank projection / global-state loading (audit defect D2)."""

import torch
import torch.nn as nn

from rank_allocation.LoRa_rank_projection import (
    load_global_state,
    project_tensor_to_rank,
)


class _LoRALayer(nn.Module):
    def __init__(self, out_f, in_f, rank):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(rank, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, rank))


class _Model(nn.Module):
    """Two LoRA keys ('layer.A', 'layer.B') plus a plain non-LoRA weight."""

    def __init__(self, out_f=10, in_f=20, rank=4):
        super().__init__()
        self.layer = _LoRALayer(out_f, in_f, rank)
        self.head = nn.Linear(in_f, out_f)


def _factors(out_f, in_f, rank, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(rank, in_f, generator=g) * scale
    b = torch.randn(out_f, rank, generator=g) * scale
    return a, b


def _global(a, b, model):
    state = {k: v.clone() for k, v in model.state_dict().items()}
    state["layer.A"] = a
    state["layer.B"] = b
    return state


def _delta(model):
    sd = model.state_dict()
    return sd["layer.B"] @ sd["layer.A"]


def _legacy_project(t, target_rank, rank_dim):
    """The pre-fix projection: SVD, keep Vh rows only, drop the singular values."""
    mat = t.float() if rank_dim == 0 else t.float().t()
    _, _, vh = torch.linalg.svd(mat, full_matrices=False)
    out = vh[:target_rank, :]
    return out if rank_dim == 0 else out.t()


# --- B1: project_tensor_to_rank must keep the singular values ----------------

def test_projection_is_scale_equivariant():
    """D2: Vh has orthonormal rows, so the old output ignored input magnitude."""
    t, _ = _factors(8, 20, 8, seed=0)
    small = project_tensor_to_rank(t, 4, 0)
    large = project_tensor_to_rank(t * 100.0, 4, 0)
    assert torch.allclose(large.abs(), small.abs() * 100.0, atol=1e-2, rtol=1e-4)


def test_projection_norm_is_not_sqrt_target_rank():
    t, _ = _factors(8, 20, 8, seed=1)
    t = t * 50.0
    for r in (2, 4, 6):
        out = project_tensor_to_rank(t, r, 0)
        assert abs(out.norm().item() - r ** 0.5) > 1.0, (
            f"output norm is still sqrt({r}), scale was destroyed"
        )


def test_projection_keeps_top_singular_mass():
    t, _ = _factors(8, 20, 8, seed=2)
    s = torch.linalg.svdvals(t.float())
    for r in (2, 4, 6):
        out = project_tensor_to_rank(t, r, 0)
        expected = torch.sqrt((s[:r] ** 2).sum())
        assert torch.allclose(out.norm(), expected, atol=1e-4)


def test_projection_rank_dim_one_matches_transpose():
    t, _ = _factors(8, 20, 8, seed=3)
    out = project_tensor_to_rank(t.t().contiguous(), 4, 1)
    assert out.shape == (20, 4)
    assert torch.allclose(out.norm(), project_tensor_to_rank(t, 4, 0).norm(), atol=1e-4)


# --- B2: load_global_state must pair A/B and refactorise the delta -----------

def test_same_shape_keys_are_copied_verbatim():
    model = _Model(rank=4)
    a, b = _factors(10, 20, 4, seed=4)
    load_global_state(model, _global(a, b, model))
    assert torch.allclose(model.state_dict()["layer.A"], a)
    assert torch.allclose(model.state_dict()["layer.B"], b)


def test_roundtrip_is_near_identity_when_target_rank_covers_the_delta():
    """Global carries a rank-4 delta padded to rank 8; local rank 4 recovers it."""
    model = _Model(rank=4)
    a, b = _factors(10, 20, 4, seed=5)
    a8 = torch.cat([a, torch.zeros(4, 20)], dim=0)
    b8 = torch.cat([b, torch.zeros(10, 4)], dim=1)
    load_global_state(model, _global(a8, b8, model))
    assert torch.allclose(_delta(model), b @ a, atol=1e-4)


def test_lower_rank_gives_the_best_rank_r_approximation():
    model = _Model(rank=3)
    a, b = _factors(10, 20, 8, seed=6)
    target = b @ a
    load_global_state(model, _global(a, b, model))
    got = _delta(model)

    u, s, vh = torch.linalg.svd(target, full_matrices=False)
    best = (u[:, :3] * s[:3]) @ vh[:3, :]
    assert torch.allclose(got, best, atol=1e-4)

    legacy = _legacy_project(b, 3, 1) @ _legacy_project(a, 3, 0)
    assert (got - target).norm() < (legacy - target).norm()


def test_delta_magnitude_is_not_a_function_of_target_rank_alone():
    """The specific D2 signature: old output norm was sqrt(r), scale-independent."""
    norms = []
    for scale in (1.0, 100.0):
        model = _Model(rank=3)
        a, b = _factors(10, 20, 8, seed=7)
        load_global_state(model, _global(a * scale, b, model))
        norms.append(_delta(model).norm().item())
    assert norms[1] / norms[0] > 50.0, f"scale information destroyed: {norms}"

    model = _Model(rank=3)
    a, b = _factors(10, 20, 8, seed=7)
    load_global_state(model, _global(a, b, model))
    assert abs(model.state_dict()["layer.A"].norm().item() - 3 ** 0.5) > 1e-2


def test_expansion_to_higher_rank_preserves_the_update():
    model = _Model(rank=6)
    a, b = _factors(10, 20, 2, seed=8)
    load_global_state(model, _global(a, b, model))
    sd = model.state_dict()
    assert sd["layer.A"].shape == (6, 20)
    assert sd["layer.B"].shape == (10, 6)
    assert torch.allclose(_delta(model), b @ a, atol=1e-4)


def test_non_lora_keys_are_untouched():
    model = _Model(rank=3)
    before_bias = model.state_dict()["head.bias"].clone()
    a, b = _factors(10, 20, 8, seed=9)
    state = _global(a, b, model)
    new_weight = torch.randn(10, 20)
    state["head.weight"] = new_weight
    state["head.bias"] = torch.randn(7)          # wrong shape -> must be ignored
    load_global_state(model, state)
    sd = model.state_dict()
    assert torch.allclose(sd["head.weight"], new_weight)
    assert torch.allclose(sd["head.bias"], before_bias)


def test_unpaired_lora_key_still_uses_elementwise_projection():
    model = _Model(rank=3)
    a, _ = _factors(10, 20, 8, seed=10)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    state["layer.A"] = a                      # B deliberately left at local shape
    load_global_state(model, state)
    assert model.state_dict()["layer.A"].shape == (3, 20)
