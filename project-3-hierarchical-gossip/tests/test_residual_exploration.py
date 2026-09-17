import numpy as np
import torch

from experiments.explore_residual_protocol import advance_global, stratified_holdout, tree_broadcast_head


def test_zero_residual_preserves_retained_model_independent_of_client_capacity():
    rng = torch.Generator().manual_seed(41)
    b = torch.randn(23, 16, generator=rng, dtype=torch.float64)
    b -= b.mean(0, keepdim=True)
    a = torch.randn(16, 32, generator=rng, dtype=torch.float64)
    delta = b @ a
    for gain in [1., 5., 15.]:
        state, result, error = advance_global(delta, torch.zeros_like(delta), gain)
        torch.testing.assert_close(result, delta, atol=1e-11, rtol=1e-11)
        assert error < 1e-24


def test_holdout_is_stratified_disjoint_exhaustive_and_deterministic():
    labels = np.repeat(np.arange(100), 500)
    fit, val = stratified_holdout(labels)
    assert len(fit) == 45000 and len(val) == 5000
    assert not set(fit).intersection(val)
    assert sorted(fit + val) == list(range(50000))
    assert np.all(np.bincount(labels[val]) == 50)
    assert (fit, val) == stratified_holdout(labels)


def test_broadcast_copies_only_on_graph_edges_and_counts_payload():
    head = torch.arange(24).float().reshape(4, 6)
    neighbors = {0: [1], 1: [0, 2], 2: [1]}
    views, ledger = tree_broadcast_head(head, neighbors, 2)
    assert set(views) == set(neighbors)
    for value in views.values():
        torch.testing.assert_close(value, head)
        assert value.data_ptr() != head.data_ptr()
    assert ledger['messages'] == 2 and ledger['bytes'] == 2 * head.numel() * 4
    assert all(edge['receiver'] in neighbors[edge['sender']] for edge in ledger['edges'])
