"""Tests for the decentralized runner (matrix-weighted delta-W gossip)."""

import numpy as np
import pytest
import torch

from src.federated.hierarchical import two_tier_mixing
from src.federated.merge import lora_to_delta
from src.federated.mixing import build_topology, metropolis_hastings
from src.federated.runner import DecentralizedRunner

ALPHA = 32.0
OUT, IN = 6, 8


class _Stub:
    """Client holding a lora state; train() is a no-op unless a step is given."""

    def __init__(self, cid, rank, seed, domain=0, accuracy=0.5):
        g = torch.Generator().manual_seed(seed)
        self.client_id, self.domain_id, self._acc = cid, domain, accuracy
        self.state = {"fc": {"A": torch.randn(rank, IN, generator=g),
                             "B": torch.randn(OUT, rank, generator=g)}}
        self.trained = 0

    def train(self):
        self.trained += 1

    def evaluate(self):
        return {"accuracy": self._acc}

    def get_lora_state(self):
        return {k: {"A": v["A"].clone(), "B": v["B"].clone()} for k, v in self.state.items()}

    def set_lora_state(self, state):
        self.state = state


def _mean_delta(states):
    return torch.stack([lora_to_delta(s, ALPHA)["fc"] for s in states]).mean(dim=0)


def _uniform(n):
    return lambda r: np.full((n, n), 1.0 / n)


def _mh_ring(n):
    w = metropolis_hastings(build_topology(list(range(n)), "ring"))
    return lambda r: w


# --- construction and validation --------------------------------------------

def test_rejects_bad_construction():
    c = [_Stub(0, 2, 0), _Stub(1, 2, 1)]
    with pytest.raises(ValueError, match="at least one"):
        DecentralizedRunner([], _uniform(0), {}, ALPHA)
    with pytest.raises(ValueError, match="target_ranks missing"):
        DecentralizedRunner(c, _uniform(2), {0: 2}, ALPHA)
    with pytest.raises(ValueError, match="alpha"):
        DecentralizedRunner(c, _uniform(2), {0: 2, 1: 2}, 0.0)
    with pytest.raises(ValueError, match=">= 1"):
        DecentralizedRunner(c, _uniform(2), {0: 0, 1: 2}, ALPHA)
    with pytest.raises(ValueError, match="unique"):
        DecentralizedRunner([_Stub(0, 2, 0), _Stub(0, 2, 1)], _uniform(2), {0: 2}, ALPHA)


def test_rejects_a_mixing_matrix_that_is_not_doubly_stochastic():
    c = [_Stub(0, 2, 0), _Stub(1, 2, 1)]
    bad = lambda r: np.array([[0.5, 0.5], [0.0, 1.0]])   # row-stochastic only
    r = DecentralizedRunner(c, bad, {0: 2, 1: 2}, ALPHA)
    with pytest.raises(ValueError, match="doubly stochastic"):
        r.gossip_round(0, [x.get_lora_state() for x in c])
    wrong_shape = lambda r: np.eye(3)
    r = DecentralizedRunner(c, wrong_shape, {0: 2, 1: 2}, ALPHA)
    with pytest.raises(ValueError, match="must be"):
        r.gossip_round(0, [x.get_lora_state() for x in c])


# --- the invariant that matters: mean preservation --------------------------

@pytest.mark.parametrize("mixing", ["uniform", "ring"])
def test_mean_delta_is_preserved_when_the_target_rank_is_sufficient(mixing):
    """Without truncation loss, one gossip round must leave the network-mean
    delta exactly where it was -- the doubly stochastic guarantee, carried
    through the delta-W merge."""
    n = 4
    clients = [_Stub(i, 2, i) for i in range(n)]
    fn = _uniform(n) if mixing == "uniform" else _mh_ring(n)
    # each mixed delta is a combination of n rank-2 deltas: rank <= 8 >= min(6, 8)
    runner = DecentralizedRunner(clients, fn, {i: 6 for i in range(n)}, ALPHA)
    states = [c.get_lora_state() for c in clients]
    before = _mean_delta(states)
    new_states, diag = runner.gossip_round(0, states)
    after = _mean_delta(new_states)
    assert torch.allclose(before, after, atol=1e-4)
    assert diag["max_tail_mass"] < 1e-9


def test_uniform_mixing_reaches_consensus_in_one_round():
    n = 5
    clients = [_Stub(i, 2, i) for i in range(n)]
    runner = DecentralizedRunner(clients, _uniform(n), {i: 6 for i in range(n)}, ALPHA)
    new_states, diag = runner.gossip_round(0, [c.get_lora_state() for c in clients])
    assert diag["consensus_distance"] < 1e-8


def test_heterogeneous_ranks_merge_and_each_client_keeps_its_own():
    clients = [_Stub(0, 2, 0), _Stub(1, 4, 1), _Stub(2, 1, 2)]
    ranks = {0: 2, 1: 4, 2: 1}
    runner = DecentralizedRunner(clients, _uniform(3), ranks, ALPHA)
    new_states, _ = runner.gossip_round(0, [c.get_lora_state() for c in clients])
    for c, s in zip(clients, new_states):
        assert s["fc"]["A"].shape[0] == ranks[c.client_id]


# --- truncation and error feedback ------------------------------------------

def test_truncation_is_measured_as_tail_mass():
    n = 4
    clients = [_Stub(i, 3, i) for i in range(n)]      # mixed rank up to 6
    runner = DecentralizedRunner(clients, _uniform(n), {i: 1 for i in range(n)}, ALPHA)
    _, diag = runner.gossip_round(0, [c.get_lora_state() for c in clients])
    assert 0.0 < diag["mean_tail_mass"] < 1.0
    assert diag["max_tail_mass"] >= diag["mean_tail_mass"]


def test_tail_mass_is_bounded_by_one_minus_r_over_R():
    """Lemma 2 of the analysis: ||C_r(X) - X||^2 <= (1 - r/min(m, n)) ||X||^2."""
    n = 3
    clients = [_Stub(i, 6, i) for i in range(n)]
    for r in (1, 2, 4):
        runner = DecentralizedRunner(clients, _uniform(n), {i: r for i in range(n)}, ALPHA)
        _, diag = runner.gossip_round(0, [c.get_lora_state() for c in clients])
        assert diag["max_tail_mass"] <= 1 - r / min(OUT, IN) + 1e-9


def test_error_feedback_conserves_the_virtual_mean():
    """Lemma 4: with feedback, the mean of (state + memory) follows exact
    averaging, so across a round with no training it is preserved exactly even
    though every stored state is truncated."""
    n = 4
    clients = [_Stub(i, 3, i) for i in range(n)]
    runner = DecentralizedRunner(clients, _mh_ring(n), {i: 1 for i in range(n)}, ALPHA,
                                 error_feedback=True)
    states = [c.get_lora_state() for c in clients]
    virtual_before = _mean_delta(states)            # memory is zero at t=0
    new_states, _ = runner.gossip_round(0, states)
    virtual_after = torch.stack([
        lora_to_delta(s, ALPHA)["fc"] + runner._memory[c.client_id]["fc"]
        for c, s in zip(clients, new_states)
    ]).mean(dim=0)
    assert torch.allclose(virtual_before, virtual_after, atol=1e-4)

    # and without feedback the stored mean drifts by the compression bias
    plain = DecentralizedRunner(clients, _mh_ring(n), {i: 1 for i in range(n)}, ALPHA)
    plain_states, _ = plain.gossip_round(0, states)
    assert not torch.allclose(virtual_before, _mean_delta(plain_states), atol=1e-4)


def test_error_feedback_memory_is_exactly_the_residual():
    n = 2
    clients = [_Stub(i, 4, i) for i in range(n)]
    runner = DecentralizedRunner(clients, _uniform(n), {i: 1 for i in range(n)}, ALPHA,
                                 error_feedback=True)
    states = [c.get_lora_state() for c in clients]
    mixed = _mean_delta(states)                    # uniform mixing: every y_i is the mean
    new_states, _ = runner.gossip_round(0, states)
    for c, s in zip(clients, new_states):
        x = lora_to_delta(s, ALPHA)["fc"]
        assert torch.allclose(runner._memory[c.client_id]["fc"], mixed - x, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_runner_returns_factors_in_the_clients_own_dtype(dtype):
    """The mixed delta is computed in a float32/float64 working dtype; the
    factors handed back must be in the dtype the client stores, exactly as
    merge_states does per layer. float16 in must not come back as float32."""
    n = 3
    clients = [_Stub(i, 2, i) for i in range(n)]
    for c in clients:
        c.state = {k: {"A": v["A"].to(dtype), "B": v["B"].to(dtype)} for k, v in c.state.items()}
    runner = DecentralizedRunner(clients, _uniform(n), {i: 4 for i in range(n)}, ALPHA)
    new, diag = runner.gossip_round(0, [c.get_lora_state() for c in clients])
    assert all(s["fc"]["A"].dtype == dtype and s["fc"]["B"].dtype == dtype for s in new)
    assert np.isfinite(diag["mean_tail_mass"]) and np.isfinite(diag["consensus_distance"])
    cs = runner.consensus_state(4)
    assert cs["fc"]["A"].dtype == dtype


def test_runner_refuses_adapters_on_different_devices():
    """A raw torch.stack device error is not an explanation."""
    n = 2
    clients = [_Stub(i, 2, i) for i in range(n)]
    runner = DecentralizedRunner(clients, _uniform(n), {i: 2 for i in range(n)}, ALPHA)
    states = [c.get_lora_state() for c in clients]
    states[1] = {k: {"A": v["A"].to("meta"), "B": v["B"].to("meta")} for k, v in states[1].items()}
    with pytest.raises(ValueError, match="different devices"):
        runner.gossip_round(0, states)


# --- communication accounting ----------------------------------------------

def test_messages_and_floats_count_nonzero_offdiagonal_entries():
    clients = [_Stub(0, 2, 0), _Stub(1, 4, 1), _Stub(2, 1, 2)]
    runner = DecentralizedRunner(clients, _mh_ring(3), {0: 2, 1: 4, 2: 1}, ALPHA)
    _, diag = runner.gossip_round(0, [c.get_lora_state() for c in clients])
    assert diag["messages"] == 6                    # ring of 3 = complete graph, 3 edges x 2 directions
    per = {0: 2 * (IN + OUT), 1: 4 * (IN + OUT), 2: 1 * (IN + OUT)}
    assert diag["floats"] == 2 * sum(per.values())  # each client's factors sent to its 2 neighbours


def test_two_tier_schedule_sends_fewer_messages_between_bridges():
    assign = {i: i // 3 for i in range(9)}
    clients = [_Stub(i, 2, i, domain=assign[i]) for i in range(9)]
    fn = lambda r: two_tier_mixing(assign, r, bridge_every=3)
    runner = DecentralizedRunner(clients, fn, {i: 6 for i in range(9)}, ALPHA)
    states = [c.get_lora_state() for c in clients]
    _, quiet = runner.gossip_round(0, states)
    _, bridge = runner.gossip_round(2, states)
    assert quiet["messages"] == 9 * 2               # 3 clusters x K_3 (3 edges x 2)
    assert bridge["messages"] > quiet["messages"]


# --- full loop ---------------------------------------------------------------

def test_run_executes_rounds_and_logs_history():
    n = 3
    clients = [_Stub(i, 2, i, domain=i % 2, accuracy=0.1 * (i + 1)) for i in range(n)]
    runner = DecentralizedRunner(clients, _mh_ring(n), {i: 4 for i in range(n)}, ALPHA)
    h = runner.run(4)
    assert all(c.trained == 4 for c in clients)
    for key in ("rounds", "avg_accuracy", "per_domain_accuracy", "per_client_accuracy",
                "messages_per_round", "floats_per_round", "mean_tail_mass",
                "max_tail_mass", "consensus_distance"):
        assert len(h[key]) == 4, key
    assert h["rounds"] == [0, 1, 2, 3]
    assert np.isclose(h["avg_accuracy"][0], 0.2)
    assert set(h["per_domain_accuracy"][0]) == {0, 1}


def test_consensus_state_is_the_uniform_merge_at_the_requested_rank():
    n = 3
    clients = [_Stub(i, 2, i) for i in range(n)]
    runner = DecentralizedRunner(clients, _uniform(n), {i: 2 for i in range(n)}, ALPHA)
    cs = runner.consensus_state(6)
    assert cs["fc"]["A"].shape[0] == 6
    got = (ALPHA / 6) * (cs["fc"]["B"] @ cs["fc"]["A"])
    want = _mean_delta([c.get_lora_state() for c in clients])
    assert torch.allclose(got, want, atol=1e-4)
