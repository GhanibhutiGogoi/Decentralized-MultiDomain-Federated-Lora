"""Numerical verification of every lemma in
docs/research/2026-09-03-project3-convergence-analysis.md.

Each test names the lemma it checks. These are not proofs -- the proofs are in
the document -- but they pin the inequalities to the code that the document
claims to describe, so a change that breaks an assumption fails here.
"""

import numpy as np
import pytest
import torch

from src.federated.hierarchical import affinity_mixing, two_tier_mixing, window_product
from src.federated.merge import factorize_delta
from src.federated.mixing import build_topology, metropolis_hastings, spectral_gap
from src.federated.runner import DecentralizedRunner

ALPHA = 32.0


def _J(n):
    return np.full((n, n), 1.0 / n)


def _mixers(n=12):
    ring = metropolis_hastings(build_topology(list(range(n)), "ring"))
    star = metropolis_hastings(build_topology(list(range(n)), "star"))
    aff = affinity_mixing(np.random.default_rng(0).normal(size=(n, n)),
                          build_topology(list(range(n)), "ring"), tau=0.7, w_min=0.15)
    assign = {i: i // 4 for i in range(n)}
    tier = window_product(lambda r: two_tier_mixing(assign, r, bridge_every=3), 0, 3)
    return {"ring": ring, "star": star, "affinity": aff, "two_tier_window": tier}


# --- Lemma 1: mixing contracts the consensus error at rate |lambda_2| ------------

@pytest.mark.parametrize("name", ["ring", "star", "affinity", "two_tier_window"])
@pytest.mark.parametrize("seed", range(5))
def test_lemma1_mixing_contraction(name, seed):
    w = _mixers()[name]
    n = w.shape[0]
    lam2 = 1.0 - spectral_gap(w)
    x = np.random.default_rng(seed).normal(size=(n, 7))
    lhs = np.linalg.norm((w - _J(n)) @ x)
    rhs = lam2 * np.linalg.norm((np.eye(n) - _J(n)) @ x)
    assert lhs <= rhs + 1e-10


@pytest.mark.parametrize("name", ["ring", "star", "affinity"])
def test_lemma1_bound_is_tight_on_the_second_eigenvector(name):
    w = _mixers()[name]
    vals, vecs = np.linalg.eigh(w)
    order = np.argsort(-np.abs(vals))
    v2 = vecs[:, order[1]]
    lam2 = abs(vals[order[1]])
    assert np.isclose(np.linalg.norm((w - _J(w.shape[0])) @ v2), lam2, atol=1e-10)


def test_lemma1_JW_equals_WJ_equals_J():
    for w in _mixers().values():
        n = w.shape[0]
        assert np.allclose(_J(n) @ w, _J(n), atol=1e-12)
        assert np.allclose(w @ _J(n), _J(n), atol=1e-12)


# --- Lemma 2: truncated SVD is a delta-contractive compressor -------------------

@pytest.mark.parametrize("shape", [(6, 8), (8, 6), (20, 5), (12, 12)])
@pytest.mark.parametrize("r", [1, 2, 4])
@pytest.mark.parametrize("seed", range(3))
def test_lemma2_compressor_bound_and_exact_tail_identity(shape, r, seed):
    m, n = shape
    x = torch.randn(m, n, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    f = factorize_delta(x, r, ALPHA)
    cx = (ALPHA / r) * (f["B"] @ f["A"])
    err = torch.sum((cx - x) ** 2).item()
    sv = torch.linalg.svdvals(x)
    tail = torch.sum(sv[r:] ** 2).item()
    assert np.isclose(err, tail, rtol=1e-8, atol=1e-10)             # exact identity
    assert err <= (1 - r / min(m, n)) * torch.sum(x ** 2).item() + 1e-9   # worst-case delta


def test_lemma2_bound_is_tight_for_a_flat_spectrum():
    m, r = 6, 2
    q, _ = torch.linalg.qr(torch.randn(m, m, dtype=torch.float64))
    x = q                                        # all singular values equal 1
    f = factorize_delta(x, r, ALPHA)
    cx = (ALPHA / r) * (f["B"] @ f["A"])
    err = torch.sum((cx - x) ** 2).item()
    assert np.isclose(err, (1 - r / m) * torch.sum(x ** 2).item(), rtol=1e-8)


def test_lemma2_is_the_metric_projection_onto_rank_r():
    """Eckart-Young: no other rank-r matrix is closer."""
    x = torch.randn(7, 9, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    f = factorize_delta(x, 3, ALPHA)
    best = torch.sum(((ALPHA / 3) * (f["B"] @ f["A"]) - x) ** 2).item()
    g = torch.Generator().manual_seed(5)
    for _ in range(300):
        rival = torch.randn(7, 3, generator=g, dtype=torch.float64) @ torch.randn(3, 9, generator=g, dtype=torch.float64)
        assert torch.sum((rival - x) ** 2).item() >= best - 1e-9


# --- Lemma 3: two-tier products are symmetric doubly stochastic with a gap ------

def test_lemma3_product_of_doubly_stochastic_is_doubly_stochastic():
    ws = list(_mixers().values())
    p = ws[0] @ ws[1] @ ws[2]
    assert np.allclose(p.sum(0), 1) and np.allclose(p.sum(1), 1) and (p >= -1e-12).all()


def test_lemma3_sandwich_is_symmetric():
    a = _mixers()["ring"]
    b = _mixers()["star"]
    s = a @ b @ a
    assert np.allclose(s, s.T, atol=1e-12)


def test_lemma3_two_tier_window_has_positive_gap_iff_bridge_in_window():
    assign = {i: i // 4 for i in range(12)}
    fn = lambda r: two_tier_mixing(assign, r, bridge_every=4)
    assert spectral_gap(window_product(fn, 0, 3)) < 1e-12     # rounds 0-2: no bridge
    assert spectral_gap(window_product(fn, 0, 4)) > 0.0       # rounds 0-3: bridge at 3


def test_lemma3_eigenvalue_one_multiplicity_equals_number_of_clusters_between_bridges():
    assign = {i: i // 4 for i in range(12)}
    w = two_tier_mixing(assign, 0, bridge_every=10)
    vals = np.linalg.eigvalsh(w)
    assert np.sum(np.isclose(vals, 1.0, atol=1e-10)) == 3


# --- Lemma 4: mean dynamics, with and without feedback ---------------------------

def _clients(n, rank, seed0=0):
    class C:
        def __init__(self, i):
            g = torch.Generator().manual_seed(seed0 + i)
            self.client_id, self.domain_id = i, 0
            self.state = {"fc": {"A": torch.randn(rank, 8, generator=g, dtype=torch.float64),
                                 "B": torch.randn(6, rank, generator=g, dtype=torch.float64)}}
        def train(self): pass
        def evaluate(self): return {"accuracy": 0.0}
        def get_lora_state(self): return {k: {"A": v["A"].clone(), "B": v["B"].clone()} for k, v in self.state.items()}
        def set_lora_state(self, s): self.state = s
    return [C(i) for i in range(n)]


def _delta(state):
    r = state["fc"]["A"].shape[0]
    return (ALPHA / r) * (state["fc"]["B"] @ state["fc"]["A"])


def test_lemma4_without_feedback_the_mean_shifts_by_the_mean_compression_error():
    n = 5
    cl = _clients(n, 3)
    w = metropolis_hastings(build_topology(list(range(n)), "ring"))
    runner = DecentralizedRunner(cl, lambda r: w, {i: 1 for i in range(n)}, ALPHA)
    states = [c.get_lora_state() for c in cl]
    deltas = torch.stack([_delta(s) for s in states])
    y = torch.einsum("ij,jab->iab", torch.tensor(w), deltas)     # mixed, pre-truncation
    new, _ = runner.gossip_round(0, states)
    x = torch.stack([_delta(s) for s in new])
    e_bar = (x - y).mean(dim=0)
    assert torch.allclose(x.mean(dim=0), deltas.mean(dim=0) + e_bar, atol=1e-10)
    assert torch.norm(e_bar) > 1e-3                              # the bias is real


def test_lemma4_with_feedback_the_virtual_mean_is_exactly_conserved_over_many_rounds():
    n = 5
    cl = _clients(n, 3)
    w = metropolis_hastings(build_topology(list(range(n)), "ring"))
    runner = DecentralizedRunner(cl, lambda r: w, {i: 1 for i in range(n)}, ALPHA,
                                 error_feedback=True)
    states = [c.get_lora_state() for c in cl]
    v0 = torch.stack([_delta(s) for s in states]).mean(dim=0)
    for t in range(20):
        states, _ = runner.gossip_round(t, states)
    v = torch.stack([_delta(s) + runner._memory[c.client_id]["fc"]
                     for c, s in zip(cl, states)]).mean(dim=0)
    assert torch.allclose(v, v0, atol=1e-9)


# --- Lemma 5: the consensus recursion ---------------------------------------------

@pytest.mark.parametrize("name", ["ring", "affinity"])
def test_lemma5_one_step_consensus_recursion_holds(name):
    """sqrt(Xi^{t+1}) <= (1 - rho) sqrt(Xi^t) + ||(I - J) E^t||, with E^t the
    truncation error, checked on the runner's own gossip step (eta = 0)."""
    n = 12
    w = _mixers()[name]
    lam2 = 1.0 - spectral_gap(w)
    cl = _clients(n, 3, seed0=100)
    runner = DecentralizedRunner(cl, lambda r: w, {i: 2 for i in range(n)}, ALPHA)
    states = [c.get_lora_state() for c in cl]
    for t in range(8):
        X = torch.stack([_delta(s) for s in states])
        xi_t = torch.sum((X - X.mean(0)) ** 2).item()
        Y = torch.einsum("ij,jab->iab", torch.tensor(w), X)
        states, _ = runner.gossip_round(t, states)
        Xn = torch.stack([_delta(s) for s in states])
        E = Xn - Y
        xi_next = torch.sum((Xn - Xn.mean(0)) ** 2).item()
        e_dev = torch.sum((E - E.mean(0)) ** 2).item()
        assert np.sqrt(xi_next) <= lam2 * np.sqrt(xi_t) + np.sqrt(e_dev) + 1e-8


def test_lemma5_without_compression_the_recursion_is_geometric():
    n = 10
    w = _mixers(n)["ring"]
    lam2 = 1.0 - spectral_gap(w)
    x = np.random.default_rng(1).normal(size=(n, 4))
    xi = [np.sum((x - x.mean(0)) ** 2)]
    for _ in range(30):
        x = w @ x
        xi.append(np.sum((x - x.mean(0)) ** 2))
    for a, b in zip(xi, xi[1:]):
        assert b <= lam2 ** 2 * a + 1e-12


# --- Lemma 6: the self-weight floor -----------------------------------------------

def test_lemma6_floor_shifts_eigenvalues_affinely():
    n = 9
    a = np.random.default_rng(2).normal(size=(n, n))
    nb = build_topology(list(range(n)), "ring")
    w0 = affinity_mixing(a, nb, tau=1.0, w_min=0.0)
    for w_min in (0.2, 0.5):
        w = affinity_mixing(a, nb, tau=1.0, w_min=w_min)
        ev0 = np.sort(np.linalg.eigvalsh(w0))
        ev = np.sort(np.linalg.eigvalsh(w))
        assert np.allclose(ev, w_min + (1 - w_min) * ev0, atol=1e-10)
        assert (np.diag(w) >= w_min - 1e-12).all()


# --- Theorem B vs Theorem A, qualitatively: the floor and what feedback does -------

def _quadratic_run(n, target_rank, error_feedback, rounds=400, eta=0.3, seed=0, star_rank=2):
    """Each client minimises f_i(X) = 1/2 ||X - M_i||^2 by projected gradient
    on rank-`target_rank`, then gossips. Returns ||x_bar - M_bar||^2 at the end
    and the irreducible rank-r error of M_bar itself."""
    g = torch.Generator().manual_seed(seed)
    M = [torch.randn(6, star_rank, generator=g, dtype=torch.float64) @
         torch.randn(star_rank, 8, generator=g, dtype=torch.float64) for _ in range(n)]
    Mbar = torch.stack(M).mean(0)

    class C:
        def __init__(self, i):
            self.client_id, self.domain_id, self.i = i, 0, i
            self.state = {"fc": factorize_delta(torch.zeros(6, 8, dtype=torch.float64), target_rank, ALPHA)}
        def train(self):
            X = _delta(self.state)
            X = X - eta * (X - M[self.i])
            self.state = {"fc": factorize_delta(X, target_rank, ALPHA)}
        def evaluate(self): return {"accuracy": 0.0}
        def get_lora_state(self): return {k: {"A": v["A"].clone(), "B": v["B"].clone()} for k, v in self.state.items()}
        def set_lora_state(self, s): self.state = s

    cl = [C(i) for i in range(n)]
    w = metropolis_hastings(build_topology(list(range(n)), "ring"))
    runner = DecentralizedRunner(cl, lambda r: w, {i: target_rank for i in range(n)}, ALPHA,
                                 error_feedback=error_feedback)
    runner.run(rounds)
    xbar = torch.stack([_delta(c.get_lora_state()) for c in cl]).mean(0)
    f = factorize_delta(Mbar, target_rank, ALPHA)
    irreducible = torch.sum(((ALPHA / target_rank) * (f["B"] @ f["A"]) - Mbar) ** 2).item()
    return torch.sum((xbar - Mbar) ** 2).item(), irreducible


def test_theorem_rank_sufficient_converges_to_the_optimum():
    """When the target rank covers rank(M_bar), the tail mass vanishes near the
    optimum and both variants converge -- the LoRA hypothesis holding."""
    n = 4
    for ef in (False, True):
        err, irreducible = _quadratic_run(n, target_rank=8, error_feedback=ef)
        assert irreducible < 1e-20
        assert err < 1e-6, (ef, err)


def test_theorem_rank_deficient_has_a_floor_that_feedback_lowers():
    """When rank(M_bar) exceeds the target rank there is an unavoidable floor
    (the irreducible rank-r error). Theorem B predicts the plain protocol
    settles ABOVE it by a bias term; Theorem A predicts feedback brings the
    stored mean close to the best rank-r approximation."""
    n = 4
    plain, irreducible = _quadratic_run(n, target_rank=2, error_feedback=False)
    fed, _ = _quadratic_run(n, target_rank=2, error_feedback=True)
    assert irreducible > 1e-3
    assert plain > irreducible * 1.05, (plain, irreducible)
    assert fed < plain, (fed, plain)
    assert fed < irreducible * 1.5, (fed, irreducible)
