"""Tests for doubly-stochastic gossip mixing (audit defect D3a)."""

import numpy as np
import pytest
import torch

from src.federated.gossip import GossipProtocol
from src.federated.mixing import build_topology, metropolis_hastings, spectral_gap

SIZES = (3, 4, 5, 8, 15)
TOPOLOGIES = ('ring', 'fully_connected')


class _StubClient:
    """Minimal client for driving the real GossipProtocol.gossip_round."""

    def __init__(self, client_id, value):
        self.client_id = client_id
        self.state = {'fc': {'A': torch.full((1, 1), float(value)),
                             'B': torch.zeros(1, 1)}}

    def get_lora_state(self):
        return {k: {'A': v['A'].clone(), 'B': v['B'].clone()}
                for k, v in self.state.items()}

    def set_lora_state(self, state):
        self.state = state


def _mean(clients):
    return float(np.mean([c.state['fc']['A'].item() for c in clients]))


# --- topology ---------------------------------------------------------------

@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("n", SIZES)
def test_topology_is_symmetric_and_simple(topology, n):
    ids = list(range(n))
    nbrs = build_topology(ids, topology)
    assert set(nbrs) == set(ids)
    for i, peers in nbrs.items():
        assert i not in peers
        assert len(peers) == len(set(peers))
        for j in peers:
            assert i in nbrs[j]


def test_ring_of_two_has_no_duplicate_neighbour():
    assert build_topology([0, 1], 'ring') == {0: [1], 1: [0]}


def test_unknown_topology_raises():
    with pytest.raises(ValueError, match="topology"):
        build_topology([0, 1, 2], 'star')


# --- mixing matrix ----------------------------------------------------------

@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("n", SIZES)
def test_mixing_matrix_is_doubly_stochastic_and_symmetric(topology, n):
    w = metropolis_hastings(build_topology(list(range(n)), topology))
    assert np.allclose(w, w.T, atol=1e-12)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-12)
    assert np.allclose(w.sum(axis=0), 1.0, atol=1e-12)
    assert (w >= -1e-12).all()


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("n", SIZES)
def test_diagonal_entries_are_non_negative(topology, n):
    w = metropolis_hastings(build_topology(list(range(n)), topology))
    assert (np.diag(w) >= 0.0).all()


def test_metropolis_hastings_matches_the_closed_form_on_a_ring():
    w = metropolis_hastings(build_topology(list(range(5)), 'ring'))
    assert np.isclose(w[0, 1], 1.0 / 3.0)
    assert np.isclose(w[0, 4], 1.0 / 3.0)
    assert np.isclose(w[0, 0], 1.0 / 3.0)
    assert np.isclose(w[0, 2], 0.0)


def test_invalid_neighbour_dicts_raise():
    with pytest.raises(ValueError, match="self"):
        metropolis_hastings({0: [0, 1], 1: [0]})
    with pytest.raises(ValueError, match="duplicate"):
        metropolis_hastings({0: [1, 1], 1: [0]})
    with pytest.raises(ValueError, match="symmetric"):
        metropolis_hastings({0: [1], 1: []})


def test_row_order_follows_the_neighbour_dict():
    ids = ['c2', 'c0', 'c1']
    w = metropolis_hastings(build_topology(ids, 'ring'), client_ids=ids)
    assert w.shape == (3, 3)
    assert np.allclose(w.sum(axis=0), 1.0, atol=1e-12)


# --- the defect: mass conservation -----------------------------------------

@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_network_mean_is_conserved_over_many_rounds(topology):
    n = 15
    w = metropolis_hastings(build_topology(list(range(n)), topology))
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    start = x.mean()
    for _ in range(500):
        x = w @ x
        assert abs(x.mean() - start) < 1e-12


def test_legacy_gossip_round_does_not_conserve_the_network_mean():
    """D3a: each node averages toward one neighbour it picks -- row-stochastic only."""
    clients = [_StubClient(i, float(i)) for i in range(15)]
    protocol = GossipProtocol(clients, topology='ring', seed=42)
    start = _mean(clients)
    drift = 0.0
    for _ in range(20):
        protocol.gossip_round()
        drift = max(drift, abs(_mean(clients) - start))
    assert drift > 1e-6, f"expected the legacy rule to lose mass, drift={drift}"


# --- spectral gap and consensus --------------------------------------------

@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("n", SIZES)
def test_spectral_gap_is_strictly_positive_for_connected_graphs(topology, n):
    gap = spectral_gap(metropolis_hastings(build_topology(list(range(n)), topology)))
    assert 0.0 < gap <= 1.0


def test_spectral_gap_is_zero_for_a_disconnected_graph():
    nbrs = {0: [1], 1: [0], 2: [3], 3: [2]}
    assert spectral_gap(metropolis_hastings(nbrs)) < 1e-12


def test_spectral_gap_needs_at_least_two_nodes():
    with pytest.raises(ValueError):
        spectral_gap(np.array([[1.0]]))


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_repeated_mixing_drives_disagreement_to_zero(topology, seed):
    """Disagreement must contract, and at no worse than the |lambda_2| rate.

    Asserted against the spectral bound rather than a fixed tolerance: the ring
    contracts at |lambda_2| = 0.91 per round, so an absolute threshold like
    1e-9 after 200 rounds is satisfied or violated depending on which random
    vector you happen to draw, and passes only for some seeds.
    """
    n = 12
    rounds = 200
    w = metropolis_hastings(build_topology(list(range(n)), topology))
    lambda_2 = 1.0 - spectral_gap(w)

    x = np.random.default_rng(seed).normal(size=n)
    spread = [np.std(x)]
    for _ in range(rounds):
        x = w @ x
        spread.append(np.std(x))

    # monotone non-increasing: symmetric doubly stochastic W is a contraction
    # on the disagreement subspace
    assert all(b <= a + 1e-12 for a, b in zip(spread, spread[1:]))
    # and it converged to consensus in relative terms
    assert spread[-1] <= spread[0] * 1e-6
    # decaying no slower than the spectral rate (with slack for float error)
    assert spread[-1] <= spread[0] * lambda_2 ** rounds * 10 + 1e-15
