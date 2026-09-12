"""Scientific invariants for the direction-aware signature comparison."""

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score
import torch

from src.clustering.signatures import (
    affinity_matrix,
    cluster_from_affinity,
    signature_delta_vec,
    signature_row_norms,
)


def state_with_rows(rows):
    a = torch.tensor([[1.0, -2.0, 0.5], [0.5, 1.0, -1.0]], dtype=torch.float64)
    b = torch.zeros(6, 2, dtype=torch.float64)
    for index, row in enumerate(rows):
        b[row, index % 2] = 1.0
    return {"fc": {"A": a, "B": b}}


def test_row_signature_retains_output_location_when_spectrum_does_not():
    left, right = state_with_rows([0, 1]), state_with_rows([4, 5])
    left_delta = left["fc"]["B"] @ left["fc"]["A"]
    right_delta = right["fc"]["B"] @ right["fc"]["A"]
    torch.testing.assert_close(torch.linalg.svdvals(left_delta), torch.linalg.svdvals(right_delta))
    signatures = [signature_row_norms(state, 32) for state in (left, right)]
    assert all(np.isclose(signature.sum(), 1) for signature in signatures)
    assert signatures[0][2:].sum() == 0
    assert signatures[1][:4].sum() == 0
    assert affinity_matrix(signatures)[0, 1] == 0


def test_signatures_are_invariant_to_nonorthogonal_factor_gauge():
    original = state_with_rows([0, 2])
    q = torch.tensor([[2.0, 0.5], [0.0, 0.4]], dtype=torch.float64)
    transformed = {"fc": {"A": q @ original["fc"]["A"],
                           "B": original["fc"]["B"] @ torch.linalg.inv(q)}}
    for signature in (signature_row_norms, signature_delta_vec):
        np.testing.assert_allclose(signature(original, 32), signature(transformed, 32), atol=1e-12)


def test_heterogeneous_rank_scaling_preserves_identical_update():
    original = state_with_rows([0, 2])
    a, b = original["fc"]["A"], original["fc"]["B"]
    padded = {"fc": {"A": torch.cat([a * 2, torch.zeros_like(a)]),
                      "B": torch.cat([b, torch.zeros_like(b)], dim=1)}}
    np.testing.assert_allclose(signature_delta_vec(original, 32),
                               signature_delta_vec(padded, 32), atol=1e-12)


def test_known_separated_domains_recover_without_labels_as_input():
    states = [state_with_rows([0, 1])] * 3 + [state_with_rows([4, 5])] * 3
    for function, kind in ((signature_row_norms, "cosine"), (signature_delta_vec, "inv_l2")):
        affinities = affinity_matrix([function(state, 32) for state in states], kind=kind)
        labels = cluster_from_affinity(affinities, n_clusters=2)
        assert adjusted_rand_score([0, 0, 0, 1, 1, 1], labels) == 1.0


def test_zero_and_multilayer_signatures_remain_finite_and_ordered():
    first = state_with_rows([])
    first["other"] = {"A": torch.ones(1, 2), "B": torch.ones(2, 1)}
    reversed_state = dict(reversed(list(first.items())))
    assert np.isfinite(signature_row_norms(first, 32)).all()
    np.testing.assert_array_equal(signature_delta_vec(first, 32),
                                  signature_delta_vec(reversed_state, 32))
    zero = signature_row_norms(state_with_rows([]), 32)
    affinity = affinity_matrix([zero, zero, np.ones_like(zero)])
    assert affinity[0, 1] == 1.0
    assert affinity[0, 2] == 0.0


@pytest.mark.parametrize("value", [np.array([[1.0, 0.1], [0.2, 1.0]]),
                                   np.array([[1.0, np.nan], [np.nan, 1.0]]),
                                   np.array([[1.0, 1.5], [1.5, 1.0]])])
def test_invalid_affinity_cannot_produce_silent_cluster_result(value):
    with pytest.raises(ValueError):
        cluster_from_affinity(value, 2)
