"""Compact merge invariants against independent dense effective-update oracles.

Run these on the authorized SSH test machine, not the local workstation.
"""

import math

import pytest
import torch

from src.federated.compact_merge import merge_compact


def state(rank, seed=0, out_features=13, in_features=17, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    return {"projection": {
        "A": torch.randn(rank, in_features, generator=generator, dtype=dtype),
        "B": torch.randn(out_features, rank, generator=generator, dtype=dtype),
    }}


def effective(value, alpha):
    a, b = value["projection"]["A"], value["projection"]["B"]
    if a.shape[0] == 0:
        return torch.zeros(b.shape[0], a.shape[1], dtype=a.dtype, device=a.device)
    return (alpha / a.shape[0]) * (b @ a)


def dense_mean(states, weights, alpha):
    return sum(weight * effective(value, alpha) for value, weight in zip(states, weights)) / sum(weights)


@pytest.mark.parametrize("dtype,atol,rtol", [(torch.float32, 2e-4, 2e-5), (torch.float64, 1e-10, 1e-11)])
def test_exact_heterogeneous_merge_with_unequal_alpha_over_rank(dtype, atol, rtol):
    states = [state(2, 1, dtype=dtype), state(5, 2, dtype=dtype)]
    snapshots = [{key: value.clone() for key, value in item["projection"].items()} for item in states]
    result, diag = merge_compact(states, [2, 7], target_rank=9, alpha=32)
    torch.testing.assert_close(effective(result, 32), dense_mean(states, [2, 7], 32), atol=atol, rtol=rtol)
    assert result["projection"]["A"].dtype == dtype
    assert diag["layers"]["projection"]["core_shape"] == [7, 7]
    assert diag["relative_discarded_energy"] == 0
    assert not diag["dense_effective_updates_materialized"]
    for original, before in zip(states, snapshots):
        for name in ("A", "B"):
            assert torch.equal(original["projection"][name], before[name])


def test_best_rank_approximation_and_discarded_energy_match_dense_svd():
    states = [state(3, 11), state(4, 12), state(2, 13)]
    dense = dense_mean(states, [1, 2, 4], 16)
    u, singular, vh = torch.linalg.svd(dense, full_matrices=False)
    target = (u[:, :4] * singular[:4]) @ vh[:4]
    result, diag = merge_compact(states, [1, 2, 4], 4, 16)
    torch.testing.assert_close(effective(result, 16), target, atol=1e-10, rtol=1e-11)
    assert diag["discarded_energy"] == pytest.approx(float(singular[4:].square().sum()), rel=1e-11)
    assert diag["effective_update_energy"] == pytest.approx(float(dense.square().sum()), rel=1e-11)
    assert diag["discarded_energy"] == pytest.approx(float((dense - effective(result, 16)).square().sum()), rel=1e-11)


def test_effective_merge_is_invariant_to_invertible_nonorthogonal_gauge():
    states = [state(3, 21), state(4, 22)]
    factors = states[0]["projection"]
    gauge = torch.tensor([[2., .3, 0.], [0., .5, .2], [0., 0., 1.5]], dtype=torch.float64)
    transformed = {"projection": {"A": gauge @ factors["A"], "B": factors["B"] @ torch.linalg.inv(gauge)}}
    before, d_before = merge_compact(states, [2, 3], 4, 8)
    after, d_after = merge_compact([transformed, states[1]], [2, 3], 4, 8)
    torch.testing.assert_close(effective(before, 8), effective(after, 8), atol=1e-10, rtol=1e-11)
    assert d_after["discarded_energy"] == pytest.approx(d_before["discarded_energy"], rel=1e-11)


def test_zero_weight_and_rank_zero_are_valid_zero_contributions():
    states = [state(2, 1), state(0, 2), state(7, 3)]
    result, diag = merge_compact(states, [2, 1, 0], 4, 32)
    torch.testing.assert_close(effective(result, 32), effective(states[0], 32) * (2 / 3), atol=1e-10, rtol=1e-11)
    assert diag["layers"]["projection"]["active_concatenated_rank"] == 2


def test_zero_target_rank_discards_all_energy():
    original = state(3, 5)
    result, diag = merge_compact([original], [1], 0, 16)
    assert result["projection"]["A"].shape == (0, 17)
    assert result["projection"]["B"].shape == (13, 0)
    assert diag["relative_discarded_energy"] == 1
    assert diag["discarded_energy"] == pytest.approx(float(effective(original, 16).square().sum()), rel=1e-11)


def test_all_zero_rank_inputs_return_exact_zero_padded_output():
    result, diag = merge_compact([state(0), state(0, 1)], [1, 3], 6, 8)
    assert torch.count_nonzero(result["projection"]["A"]) == 0
    assert torch.count_nonzero(result["projection"]["B"]) == 0
    assert diag["effective_update_energy"] == diag["relative_discarded_energy"] == 0
    assert diag["layers"]["projection"]["core_shape"] == [0, 0]


def test_padding_above_feature_dimensions_and_dense_core_are_reported():
    original = state(6, out_features=3, in_features=4)
    result, diag = merge_compact([original], [1], 7, 8)
    torch.testing.assert_close(effective(result, 8), effective(original, 8), atol=1e-10, rtol=1e-11)
    assert torch.count_nonzero(result["projection"]["A"][3:]) == 0
    assert torch.count_nonzero(result["projection"]["B"][:, 3:]) == 0
    assert diag["layers"]["projection"]["core_reaches_dense_shape"]


def test_different_layer_ranks_and_mixed_client_dtypes_promote_per_layer():
    a, b = state(2, dtype=torch.float32), state(3, dtype=torch.float64)
    a["other"] = state(4, out_features=7, in_features=9, dtype=torch.float32)["projection"]
    b["other"] = state(2, 1, out_features=7, in_features=9, dtype=torch.float32)["projection"]
    result, diag = merge_compact([a, b], [1, 2], 6, 16)
    assert result["projection"]["A"].dtype == torch.float64
    assert result["other"]["A"].dtype == torch.float32
    assert diag["layers"]["other"]["core_shape"] == [6, 6]


def test_large_valid_weights_do_not_overflow_normalization():
    a, b = state(2), state(3, 1)
    result, diag = merge_compact([a, b], [1e308, 1e308], 5, 8)
    torch.testing.assert_close(effective(result, 8), dense_mean([a, b], [1, 1], 8), atol=1e-10, rtol=1e-11)
    assert diag["normalized_weights"] == [.5, .5]


def test_svd_only_receives_small_core(monkeypatch):
    # This catches accidental regression to dense feature-dimension SVD.
    original_svd = torch.linalg.svd
    observed = []
    def tracked(value, *args, **kwargs):
        observed.append(tuple(value.shape))
        return original_svd(value, *args, **kwargs)
    monkeypatch.setattr(torch.linalg, "svd", tracked)
    _, diag = merge_compact([state(2, out_features=128, in_features=192),
                             state(4, 1, out_features=128, in_features=192)], [1, 2], 4, 8)
    assert observed == [(6, 6)]
    assert not diag["layers"]["projection"]["core_reaches_dense_shape"]
    assert "not a measured peak" in diag["workspace_scope"]


@pytest.mark.parametrize("weights", [[-1, 2], [0, 0], [math.nan, 1], [math.inf, 1], [1]])
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError, match="weights"):
        merge_compact([state(2), state(3)], weights, 4, 8)


@pytest.mark.parametrize("rank", [-1, 1.5, True])
def test_invalid_target_rank_rejected(rank):
    with pytest.raises(ValueError, match="target_rank"):
        merge_compact([state(2)], [1], rank, 8)


@pytest.mark.parametrize("alpha", [0, -1, math.nan, math.inf])
def test_invalid_alpha_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        merge_compact([state(2)], [1], 4, alpha)


@pytest.mark.parametrize("fault", ["shape", "features", "nan", "dtype", "layers", "missing", "tensor", "empty_features"])
def test_invalid_states_rejected_even_when_weight_zero(fault):
    a, b = state(2), state(3)
    if fault == "shape":
        b["projection"]["B"] = torch.zeros(13, 2, dtype=torch.float64)
    elif fault == "features":
        b["projection"]["B"] = torch.zeros(14, 3, dtype=torch.float64)
    elif fault == "nan":
        b["projection"]["A"][0, 0] = math.nan
    elif fault == "dtype":
        b["projection"]["A"] = b["projection"]["A"].half()
    elif fault == "layers":
        b = {"different": b["projection"]}
    elif fault == "missing":
        del b["projection"]["A"]
    elif fault == "tensor":
        b["projection"]["A"] = [[1]]
    elif fault == "empty_features":
        b["projection"]["A"] = torch.empty(3, 0, dtype=torch.float64)
    with pytest.raises(ValueError):
        merge_compact([a, b], [1, 0], 4, 8)


def test_empty_inputs_and_empty_layer_sets():
    with pytest.raises(ValueError, match="at least one"):
        merge_compact([], [], 4, 8)
    result, diag = merge_compact([{}, {}], [1, 2], 4, 8)
    assert result == {}
    assert diag["effective_update_energy"] == diag["output_factor_bytes"] == 0
