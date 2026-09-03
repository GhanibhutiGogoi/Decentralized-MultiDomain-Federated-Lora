"""Tests for hierarchical and affinity-weighted mixing."""

import numpy as np
import pytest

from src.federated.hierarchical import (
    SINKHORN_TOL,
    affinity_mixing,
    clusters_from_assignments,
    is_doubly_stochastic,
    sinkhorn,
    two_tier_mixing,
    window_product,
)
from src.federated.mixing import build_topology, metropolis_hastings, spectral_gap


def _ds(w, atol=1e-10):
    assert is_doubly_stochastic(w, atol=atol), (w.sum(0), w.sum(1))


def _sym(w, atol=1e-12):
    assert np.allclose(w, w.T, atol=atol)


# --- sinkhorn ---------------------------------------------------------------

@pytest.mark.parametrize("n", (2, 3, 7, 15))
@pytest.mark.parametrize("seed", range(4))
def test_sinkhorn_returns_doubly_stochastic(n, seed):
    m = np.random.default_rng(seed).random((n, n)) + 0.05
    _ds(sinkhorn(m))


@pytest.mark.parametrize("seed", range(4))
def test_sinkhorn_preserves_symmetry_exactly(seed):
    m = np.random.default_rng(seed).random((6, 6)) + 0.05
    m = 0.5 * (m + m.T)
    w = sinkhorn(m)
    _ds(w)
    assert np.array_equal(w, w.T)


def test_sinkhorn_respects_the_zero_pattern():
    m = np.array([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
    w = sinkhorn(m)
    assert w[0, 2] == 0.0 and w[2, 0] == 0.0
    _ds(w)


def test_sinkhorn_rejects_bad_input():
    with pytest.raises(ValueError, match="square"):
        sinkhorn(np.ones((2, 3)))
    with pytest.raises(ValueError, match="non-negative"):
        sinkhorn(np.array([[1.0, -1.0], [1.0, 1.0]]))
    with pytest.raises(ValueError, match="finite"):
        sinkhorn(np.array([[1.0, np.nan], [1.0, 1.0]]))
    with pytest.raises(ValueError, match="positive entry"):
        sinkhorn(np.array([[1.0, 0.0], [0.0, 0.0]]))


def test_sinkhorn_raises_when_the_pattern_lacks_total_support():
    """(0, 0) is positive but lies on no positive permutation, so the
    iteration cannot reach doubly stochastic. It must say so, not return a
    near-miss."""
    m = np.array([[1.0, 1.0], [1.0, 0.0]])
    with pytest.raises(RuntimeError, match="did not reach"):
        sinkhorn(m, iters=200)


# --- affinity mixing --------------------------------------------------------

def _ring(n):
    return build_topology(list(range(n)), "ring")


@pytest.mark.parametrize("n", (3, 6, 15))
@pytest.mark.parametrize("w_min", (0.0, 0.1, 0.5))
def test_affinity_mixing_is_symmetric_doubly_stochastic(n, w_min):
    a = np.random.default_rng(n).normal(size=(n, n))
    w = affinity_mixing(a, _ring(n), tau=1.0, w_min=w_min)
    _ds(w)
    _sym(w)
    assert (np.diag(w) >= w_min - 1e-12).all()


def test_affinity_mixing_support_is_neighbours_plus_self():
    n = 8
    a = np.zeros((n, n))
    w = affinity_mixing(a, _ring(n), tau=1.0, w_min=0.0)
    nbrs = _ring(n)
    for i in range(n):
        for j in range(n):
            expected = (i == j) or (j in nbrs[i])
            assert (w[i, j] > 0) == expected


def test_higher_affinity_gets_more_weight():
    """On a ring every node has two neighbours; the one it is more similar to
    should receive the larger share."""
    n = 6
    a = np.zeros((n, n))
    a[0, 1] = a[1, 0] = 3.0      # 0 likes 1
    a[0, 5] = a[5, 0] = -3.0     # 0 dislikes 5
    w = affinity_mixing(a, _ring(n), tau=1.0, w_min=0.0)
    assert w[0, 1] > w[0, 5]
    assert w[0, 1] > 2 * w[0, 5]


def test_large_temperature_approaches_uniform_on_the_support():
    n = 6
    a = np.random.default_rng(1).normal(size=(n, n)) * 5
    w_hot = affinity_mixing(a, _ring(n), tau=1e6, w_min=0.0)
    w_flat = affinity_mixing(np.zeros((n, n)), _ring(n), tau=1.0, w_min=0.0)
    assert np.allclose(w_hot, w_flat, atol=1e-5)


def test_small_temperature_concentrates_on_the_best_neighbour():
    """Lower temperature must shift mass toward the preferred neighbour, and
    keep doing so as tau falls. Tested as a monotone trend plus a relative
    ratio, not an absolute threshold: at very low tau the pair {0, 1} becomes
    nearly disconnected from the rest and Sinkhorn crawls through the
    bottleneck, which is a property of the projection, not a bug."""
    n = 6
    a = np.zeros((n, n))
    a[0, 1] = a[1, 0] = 1.0
    ratios = []
    for tau in (2.0, 1.0, 0.5, 0.25):
        w = affinity_mixing(a, _ring(n), tau=tau, w_min=0.0)
        _ds(w)
        ratios.append(w[0, 1] / w[0, 5])
    assert all(b > a_ for a_, b in zip(ratios, ratios[1:])), ratios   # sharper as tau falls
    # The projected ratio is well below the raw kernel ratio (e^4 ~ 55 at
    # tau=0.25) because Sinkhorn's column balancing pulls mass back from the
    # over-subscribed node; ~10 is what the projection actually yields. The
    # uniform limit is 1, so the floor is set well above that, not near the
    # raw-kernel value.
    assert ratios[-1] > 5, ratios


def test_temperature_too_small_for_the_affinity_range_is_refused():
    """tau=0.01 over a spread of 1.0 is 100 nats: the kernel spans e^-100 and
    Sinkhorn crawls instead of converging. That must be a clear error naming
    tau, not a RuntimeError about iteration counts."""
    n = 6
    a = np.zeros((n, n))
    a[0, 1] = a[1, 0] = 1.0
    with pytest.raises(ValueError, match="tau=0.01 is too small"):
        affinity_mixing(a, _ring(n), tau=0.01, w_min=0.0)


def test_self_weight_floor_scales_the_spectral_gap_exactly():
    """W = w_min I + (1 - w_min) W0 shifts every eigenvalue affinely, so the
    gap shrinks by exactly (1 - w_min). This is the documented price of the
    floor and should hold to machine precision."""
    n = 10
    a = np.random.default_rng(3).normal(size=(n, n))
    g0 = spectral_gap(affinity_mixing(a, _ring(n), tau=1.0, w_min=0.0))
    for w_min in (0.1, 0.3, 0.7):
        g = spectral_gap(affinity_mixing(a, _ring(n), tau=1.0, w_min=w_min))
        assert np.isclose(g, (1 - w_min) * g0, atol=1e-10)


def test_asymmetric_affinity_is_symmetrised():
    n = 5
    a = np.random.default_rng(7).normal(size=(n, n))
    w = affinity_mixing(a, _ring(n), tau=1.0, w_min=0.0)
    _sym(w)
    _ds(w)


def test_affinity_mixing_validation():
    n = 4
    a = np.zeros((n, n))
    with pytest.raises(ValueError, match="tau"):
        affinity_mixing(a, _ring(n), tau=0.0)
    with pytest.raises(ValueError, match="w_min"):
        affinity_mixing(a, _ring(n), w_min=1.0)
    with pytest.raises(ValueError, match="shape"):
        affinity_mixing(np.zeros((3, 3)), _ring(n))
    with pytest.raises(ValueError, match="finite"):
        affinity_mixing(np.full((n, n), np.inf), _ring(n))
    with pytest.raises(ValueError, match="unique"):
        affinity_mixing(a, _ring(n), client_ids=[0, 1, 1, 2])
    with pytest.raises(ValueError, match="exactly the clients"):
        affinity_mixing(a, _ring(n), client_ids=[0, 1, 2, 9])


def test_affinity_mixing_conserves_the_mean_to_tolerance_and_contracts_at_the_spectral_rate():
    """A Sinkhorn output is doubly stochastic to within SINKHORN_TOL, so the
    mean is conserved to about tol * max|x| per round -- not to 1e-12 the way
    an analytic Metropolis-Hastings matrix conserves it. The bound asserted is
    the one the tolerance implies."""
    n = 12
    a = np.random.default_rng(5).normal(size=(n, n))
    w = affinity_mixing(a, _ring(n), tau=0.5, w_min=0.2)
    assert np.abs(w.sum(axis=0) - 1).max() <= SINKHORN_TOL
    rho = spectral_gap(w)
    assert rho > 0
    x = np.random.default_rng(6).normal(size=n)
    start, spread0, scale = x.mean(), np.std(x), np.abs(x).max()
    rounds = 300
    for _ in range(rounds):
        x = w @ x
    assert abs(x.mean() - start) <= rounds * SINKHORN_TOL * scale      # what tol implies
    assert np.std(x) <= spread0 * (1 - rho) ** rounds + 1e-12          # Lemma 1
    assert np.std(x) < spread0


# --- two-tier mixing --------------------------------------------------------

def _assign(n_clusters=3, per=4):
    return {i: i // per for i in range(n_clusters * per)}


def test_clusters_from_assignments_groups_and_sorts():
    groups = clusters_from_assignments({0: "b", 1: "a", 2: "b", 3: "a"})
    assert groups == {"a": [1, 3], "b": [0, 2]}


@pytest.mark.parametrize("round_idx", range(6))
@pytest.mark.parametrize("bridge_every", (1, 3, 5))
def test_two_tier_is_symmetric_doubly_stochastic_every_round(round_idx, bridge_every):
    w = two_tier_mixing(_assign(), round_idx, bridge_every=bridge_every)
    _ds(w)
    _sym(w)


def test_non_bridge_rounds_are_block_diagonal():
    """Between bridges no mass crosses a cluster boundary at all."""
    assign = _assign(3, 4)
    for r in (0, 1, 2, 3):          # bridge_every=5 bridges at round 4
        w = two_tier_mixing(assign, r, bridge_every=5)
        for i in range(12):
            for j in range(12):
                if assign[i] != assign[j]:
                    assert w[i, j] == 0.0


def test_bridge_rounds_connect_every_cluster():
    assign = _assign(3, 4)
    w = two_tier_mixing(assign, 4, bridge_every=5)
    for k in range(3):
        for l in range(3):
            if k != l:
                block = w[np.ix_([i for i in range(12) if assign[i] == k],
                                 [j for j in range(12) if assign[j] == l])]
                assert (block > 0).all(), f"clusters {k} and {l} did not mix"


def test_bridge_every_one_bridges_every_round():
    assign = _assign(2, 3)
    for r in range(4):
        w = two_tier_mixing(assign, r, bridge_every=1)
        assert w[0, 3] > 0


def test_single_cluster_never_bridges():
    assign = {i: 0 for i in range(5)}
    for r in range(6):
        w = two_tier_mixing(assign, r, bridge_every=2)
        _ds(w)
        assert np.allclose(w, np.full((5, 5), 0.2))   # complete-graph MH is uniform averaging


def test_singleton_clusters_are_handled():
    assign = {0: 0, 1: 1, 2: 1, 3: 2}
    w = two_tier_mixing(assign, 0, bridge_every=1)
    _ds(w)
    _sym(w)


def test_non_bridge_round_has_no_cross_cluster_gap_but_the_window_does():
    """A block-diagonal matrix has eigenvalue 1 with multiplicity K, so its
    spectral gap is exactly zero. The product over one full period is the
    object with a gap, which is what consensus depends on."""
    assign = _assign(3, 4)
    fn = lambda r: two_tier_mixing(assign, r, bridge_every=4)
    assert spectral_gap(fn(0)) < 1e-12
    p = window_product(fn, 0, 4)
    _ds(p)
    assert spectral_gap(p) > 0.05


def test_complete_intra_mixing_makes_the_window_product_equal_the_bridge_matrix():
    """Uniform within-cluster averaging is idempotent, so W_full W_intra^(L-1)
    == W_full. This is the observation that reduces the two-tier schedule to
    the L = 1 case of the convergence analysis."""
    assign = _assign(3, 4)
    fn = lambda r: two_tier_mixing(assign, r, bridge_every=5)
    p = window_product(fn, 0, 5)
    assert np.allclose(p, fn(4), atol=1e-12)


def test_transfer_weights_bias_the_bridge():
    """Raising T_01 must move more mass between clusters 0 and 1 than between
    0 and 2."""
    assign = _assign(3, 4)
    t = np.ones((3, 3))
    t[0, 1] = t[1, 0] = 4.0
    w = two_tier_mixing(assign, 0, bridge_every=1, transfer=t, bridge_tau=1.0)
    m01 = w[np.ix_(range(0, 4), range(4, 8))].sum()
    m02 = w[np.ix_(range(0, 4), range(8, 12))].sum()
    assert m01 > 2 * m02
    _ds(w)


def test_two_tier_conserves_the_mean_under_the_time_varying_schedule():
    """Mean is exact every round. Consensus across clusters is slower than it
    looks: a representative carries only 1/m of its cluster's weight, so with
    clusters of 3 the cluster means contract by exactly 2/3 per bridge. The
    round count is therefore derived from the window gap, not guessed."""
    assign = _assign(4, 3)
    period = 3
    fn = lambda r: two_tier_mixing(assign, r, bridge_every=period)
    rho = spectral_gap(window_product(fn, 0, period))
    assert np.isclose(rho, 1 / 3, atol=1e-10)              # the 2/3-per-bridge contraction
    windows = int(np.ceil(np.log(1e-8) / np.log(1 - rho)))
    x = np.random.default_rng(0).normal(size=12)
    start, spread0 = x.mean(), np.std(x)
    for r in range(windows * period):
        x = fn(r) @ x
    assert abs(x.mean() - start) < 1e-12
    assert np.std(x) <= spread0 * (1 - rho) ** windows * 10 + 1e-12
    assert np.std(x) < 1e-6


def test_two_tier_validation():
    assign = _assign(2, 2)
    with pytest.raises(ValueError, match="bridge_every"):
        two_tier_mixing(assign, 0, bridge_every=0)
    with pytest.raises(ValueError, match="transfer must have shape"):
        two_tier_mixing(assign, 0, bridge_every=1, transfer=np.ones((3, 3)))
    with pytest.raises(ValueError, match="finite"):
        two_tier_mixing(assign, 0, bridge_every=1, transfer=np.full((2, 2), np.nan))
    with pytest.raises(ValueError, match="not a member"):
        two_tier_mixing(assign, 0, bridge_every=1, representatives={0: 2, 1: 3})
    with pytest.raises(ValueError, match="unique"):
        two_tier_mixing(assign, 0, client_ids=[0, 0, 1, 2])


def test_window_product_multiplies_in_time_order():
    ws = [np.array([[0.5, 0.5], [0.5, 0.5]]), np.eye(2), np.array([[0.0, 1.0], [1.0, 0.0]])]
    p = window_product(lambda r: ws[r], 0, 3)
    assert np.allclose(p, ws[2] @ ws[1] @ ws[0])
    with pytest.raises(ValueError, match="length"):
        window_product(lambda r: ws[0], 0, 0)


def test_is_doubly_stochastic():
    assert is_doubly_stochastic(np.eye(3))
    assert is_doubly_stochastic(metropolis_hastings(_ring(5)))
    assert not is_doubly_stochastic(np.array([[0.5, 0.5], [0.0, 1.0]]))
    assert not is_doubly_stochastic(np.array([[1.5, -0.5], [-0.5, 1.5]]))
