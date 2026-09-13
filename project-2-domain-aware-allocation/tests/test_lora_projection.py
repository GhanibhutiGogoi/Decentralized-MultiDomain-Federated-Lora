"""Regression tests for heterogeneous-rank update transport."""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

PROJECT2_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT2_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT2_ROOT))

from framework.aggregation.projection import load_global_state, project_tensor_to_rank


class _Model(nn.Module):
    def __init__(self, rank, out_features=5, in_features=7):
        super().__init__()
        self.layer = nn.Module()
        self.layer.A = nn.Parameter(torch.zeros(rank, in_features))
        self.layer.B = nn.Parameter(torch.zeros(out_features, rank))
        self.head = nn.Linear(in_features, out_features)


def _factors(rank, out_features=5, in_features=7):
    generator = torch.Generator().manual_seed(47)
    return (
        torch.randn(rank, in_features, generator=generator),
        torch.randn(out_features, rank, generator=generator),
    )


@pytest.mark.parametrize("rank_dim", [0, 1])
def test_single_factor_compression_keeps_singular_energy_and_input_scale(rank_dim):
    a, _ = _factors(6)
    matrix = a if rank_dim == 0 else a.T
    projected = project_tensor_to_rank(matrix, 3, rank_dim)
    scaled = project_tensor_to_rank(matrix * 17, 3, rank_dim)
    expected_energy = torch.linalg.svdvals(matrix)[:3].square().sum()
    torch.testing.assert_close(projected.square().sum(), expected_energy)
    torch.testing.assert_close(scaled.norm(), projected.norm() * 17)


@pytest.mark.parametrize("rank_dim", [0, 1])
def test_single_factor_compression_pads_when_feature_dimension_is_small(rank_dim):
    matrix = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    if rank_dim == 1:
        matrix = matrix.T
    projected = project_tensor_to_rank(matrix, 4, rank_dim)
    assert projected.shape[rank_dim] == 4
    torch.testing.assert_close(projected.norm(), matrix.norm())


@pytest.mark.parametrize("source_rank,target_rank", [(6, 2), (2, 6), (6, 8)])
def test_paired_loading_preserves_the_best_rank_r_update(source_rank, target_rank):
    model = _Model(target_rank)
    a, b = _factors(source_rank)
    global_state = {"layer.A": a, "layer.B": b}
    load_global_state(model, global_state)

    target = b @ a
    u, singular_values, vh = torch.linalg.svd(target, full_matrices=False)
    best = (u[:, :target_rank] * singular_values[:target_rank]) @ vh[:target_rank]
    actual = model.layer.B @ model.layer.A
    torch.testing.assert_close(actual, best, atol=1e-5, rtol=1e-5)


def test_same_rank_factors_are_copied_without_changing_their_basis():
    model = _Model(3)
    a, b = _factors(3)
    head_weight = torch.ones_like(model.head.weight)
    original_bias = model.head.bias.detach().clone()
    load_global_state(model, {
        "layer.A": a,
        "layer.B": b,
        "head.weight": head_weight,
        "head.bias": torch.zeros(9),
    })
    torch.testing.assert_close(model.layer.A, a)
    torch.testing.assert_close(model.layer.B, b)
    torch.testing.assert_close(model.head.weight, head_weight)
    torch.testing.assert_close(model.head.bias, original_bias)


def test_unpaired_loading_retains_the_single_factor_fallback():
    model = _Model(2)
    a, _ = _factors(6)
    load_global_state(model, {"layer.A": a})
    torch.testing.assert_close(model.layer.A, project_tensor_to_rank(a, 2))


def test_geometry_mismatch_is_distinguished_from_rank_mismatch():
    model = _Model(2)
    a, b = _factors(6, in_features=9)
    with pytest.raises(ValueError, match="geometry mismatch"):
        load_global_state(model, {"layer.A": a, "layer.B": b})


def test_inconsistent_local_pair_is_rejected():
    model = _Model(2)
    model.layer.B = nn.Parameter(torch.zeros(5, 3))
    a, b = _factors(6)
    with pytest.raises(ValueError, match="local LoRA pair"):
        load_global_state(model, {"layer.A": a, "layer.B": b})
