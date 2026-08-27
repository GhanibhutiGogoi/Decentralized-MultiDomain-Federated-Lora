"""Tests for doubly-stochastic gossip mixing (audit defect D3a)."""

import numpy as np
import pytest
import torch

from src.federated.gossip import GossipProtocol
from src.federated.mixing import build_topology, metropolis_hastings, spectral_gap

SIZES = (3, 4, 5, 8, 15)
TOPOLOGIES = ('ring', 'fully_connected', 'star', 'path')
# star and path are IRREGULAR: their nodes have differing degrees, which is
# what makes the max(deg_i, deg_j) in the Metropolis-Hastings rule bite. On a
# regular graph that max is a no-op, so a suite of rings and cliques alone
# cannot distinguish the correct rule from the naive 1/(1+deg_i) one.
IRREGULAR = ('star', 'path')


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
        build_topology([0, 1, 2], 'hypercube')


def test_duplicate_client_ids_raise():
    with pytest.raises(ValueError, match="unique"):
        build_topology([0, 1, 1], 'ring')


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

    Both the tolerance and the round count are derived from the spectral gap
    rather than hard-coded: a fixed "1e-9 after 200 rounds" is satisfied or
    violated depending on which random vector you draw (it passed for 1 seed in
    6 on a ring), and a path graph at |lambda_2| = 0.977 needs ~1000 rounds to
    reach the same place a clique reaches in one.
    """
    n = 12
    w = metropolis_hastings(build_topology(list(range(n)), topology))
    lambda_2 = 1.0 - spectral_gap(w)
    rounds = 1 if lambda_2 <= 0 else min(int(np.ceil(np.log(1e-10) / np.log(lambda_2))), 5000)

    x = np.random.default_rng(seed).normal(size=n)
    spread = [np.std(x)]
    for _ in range(rounds):
        x = w @ x
        spread.append(np.std(x))

    # symmetric doubly stochastic W contracts the disagreement subspace
    assert all(b <= a + 1e-12 for a, b in zip(spread, spread[1:]))
    assert spread[-1] <= spread[0] * 1e-6
    assert spread[-1] <= spread[0] * lambda_2 ** rounds * 10 + 1e-15


# --- the max(deg_i, deg_j) is load-bearing, and only irregular graphs show it --

@pytest.mark.parametrize("topology", IRREGULAR)
@pytest.mark.parametrize("n", (4, 5, 8, 15))
def test_naive_degree_rule_is_not_doubly_stochastic_on_irregular_graphs(topology, n):
    """Guards the specific thing Metropolis-Hastings buys over 1/(1 + deg_i).

    Without this, reverting `max(degree[cid], degree[peer])` to `degree[cid]`
    leaves the whole suite green, because every ring and clique is regular.
    """
    neighbors = build_topology(list(range(n)), topology)
    order = sorted(neighbors)
    degree = {c: len(v) for c, v in neighbors.items()}
    assert len(set(degree.values())) > 1, "topology must be irregular for this test to bind"

    naive = np.zeros((n, n))
    for i, cid in enumerate(order):
        for peer in neighbors[cid]:
            naive[i, order.index(peer)] = 1.0 / (1.0 + degree[cid])
        naive[i, i] = 1.0 - naive[i].sum()

    assert np.allclose(naive.sum(axis=1), 1.0), "the naive rule is still row-stochastic"
    assert not np.allclose(naive.sum(axis=0), 1.0), "naive rule must break column sums"
    assert not np.allclose(naive, naive.T), "naive rule must break symmetry"

    correct = metropolis_hastings(neighbors)
    assert not np.allclose(correct, naive), "Metropolis-Hastings must differ from the naive rule"
    assert np.allclose(correct.sum(axis=0), 1.0, atol=1e-12)


@pytest.mark.parametrize("topology", IRREGULAR)
def test_naive_degree_rule_loses_mass_where_metropolis_hastings_does_not(topology):
    n = 8
    neighbors = build_topology(list(range(n)), topology)
    order = sorted(neighbors)
    degree = {c: len(v) for c, v in neighbors.items()}

    naive = np.zeros((n, n))
    for i, cid in enumerate(order):
        for peer in neighbors[cid]:
            naive[i, order.index(peer)] = 1.0 / (1.0 + degree[cid])
        naive[i, i] = 1.0 - naive[i].sum()

    correct = metropolis_hastings(neighbors)
    x0 = np.random.default_rng(0).normal(size=n)
    start = x0.mean()

    x = x0.copy()
    for _ in range(200):
        x = naive @ x
    assert abs(x.mean() - start) > 1e-6, "naive rule should drift off the network mean"

    x = x0.copy()
    for _ in range(200):
        x = correct @ x
    assert abs(x.mean() - start) < 1e-12, "Metropolis-Hastings must conserve it exactly"
