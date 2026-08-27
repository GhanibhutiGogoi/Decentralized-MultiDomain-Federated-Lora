"""Tests for the delta-W merge kernel (audit defects D3b and D6).

The comparator throughout is `GossipProtocol._pairwise_average`, the shipped
factor-space averaging, so each test states exactly what the new kernel fixes.
"""

import pytest
import torch

from src.federated.gossip import GossipProtocol
from src.federated.merge import factorize_delta, lora_to_delta, merge_states

ALPHA = 32.0


def make_state(out_f=10, in_f=20, rank=4, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return {
        'fc': {
            'A': torch.randn(rank, in_f, generator=g) * scale,
            'B': torch.randn(out_f, rank, generator=g) * scale,
        }
    }


def factor_average(state1, state2):
    """The shipped gossip mixing step, called directly (self is unused)."""
    return GossipProtocol._pairwise_average(None, state1, state2)


def rotate(state, seed=0):
    """Gauge transform: (B, A) -> (B Q^T, Q A) for orthogonal Q."""
    rotated = {}
    for layer, p in state.items():
        r = p['A'].shape[0]
        g = torch.Generator().manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(r, r, generator=g))
        rotated[layer] = {'A': q @ p['A'], 'B': p['B'] @ q.T}
    return rotated


# --- alpha/r scaling (D6) ---------------------------------------------------

def test_lora_to_delta_applies_alpha_over_rank():
    state = make_state(rank=4)
    delta = lora_to_delta(state, ALPHA)['fc']
    assert torch.allclose(delta, (ALPHA / 4) * state['fc']['B'] @ state['fc']['A'])


def test_unscaled_merge_would_differ_under_heterogeneous_ranks():
    """D6: alpha/r differs by 4x between rank 2 and rank 8; B@A alone hides that."""
    s_low, s_high = make_state(rank=2, seed=1), make_state(rank=8, seed=2)
    scaled = 0.5 * lora_to_delta(s_low, ALPHA)['fc'] + 0.5 * lora_to_delta(s_high, ALPHA)['fc']
    unscaled = 0.5 * (s_low['fc']['B'] @ s_low['fc']['A']) + 0.5 * (s_high['fc']['B'] @ s_high['fc']['A'])
    assert not torch.allclose(scaled, unscaled * ALPHA, atol=1e-3)


# --- gauge invariance (D3b) -------------------------------------------------

def test_merge_is_gauge_invariant_but_factor_averaging_is_not():
    s1, s2 = make_state(rank=4, seed=1), make_state(rank=4, seed=2)
    s1_rot = rotate(s1, seed=7)

    assert torch.allclose(
        lora_to_delta(s1, ALPHA)['fc'], lora_to_delta(s1_rot, ALPHA)['fc'], atol=1e-5
    ), "the rotation must not change the client's own update"

    merged = merge_states([s1, s2], [0.5, 0.5], target_rank=8, alpha=ALPHA)
    merged_rot = merge_states([s1_rot, s2], [0.5, 0.5], target_rank=8, alpha=ALPHA)
    d = lora_to_delta(merged, ALPHA)['fc']
    d_rot = lora_to_delta(merged_rot, ALPHA)['fc']
    assert torch.allclose(d, d_rot, atol=1e-5)

    avg = factor_average(s1, s2)['fc']
    avg_rot = factor_average(s1_rot, s2)['fc']
    assert not torch.allclose(avg['B'] @ avg['A'], avg_rot['B'] @ avg_rot['A'], atol=1e-5), (
        "factor averaging was expected to be gauge dependent"
    )


# --- heterogeneous ranks ----------------------------------------------------

def test_heterogeneous_ranks_merge_where_factor_averaging_cannot():
    s1, s2 = make_state(rank=2, seed=1), make_state(rank=6, seed=2)
    with pytest.raises(RuntimeError):
        factor_average(s1, s2)

    merged = merge_states([s1, s2], [0.5, 0.5], target_rank=8, alpha=ALPHA)
    assert merged['fc']['A'].shape == (8, 20)
    assert merged['fc']['B'].shape == (10, 8)
    dense = 0.5 * lora_to_delta(s1, ALPHA)['fc'] + 0.5 * lora_to_delta(s2, ALPHA)['fc']
    assert torch.allclose(lora_to_delta(merged, ALPHA)['fc'], dense, atol=1e-4)


# --- round trip / identity --------------------------------------------------

def test_roundtrip_exact_when_target_rank_covers_the_delta():
    state = make_state(rank=4)
    delta = lora_to_delta(state, ALPHA)['fc']
    factors = factorize_delta(delta, target_rank=4, alpha=ALPHA)
    recon = (ALPHA / 4) * factors['B'] @ factors['A']
    assert factors['A'].shape == (4, 20) and factors['B'].shape == (10, 4)
    assert torch.allclose(recon, delta, atol=1e-4)


def test_merging_identical_states_returns_that_state_delta():
    state = make_state(rank=4, seed=3)
    merged = merge_states([state] * 4, [0.25] * 4, target_rank=4, alpha=ALPHA)
    assert torch.allclose(
        lora_to_delta(merged, ALPHA)['fc'], lora_to_delta(state, ALPHA)['fc'], atol=1e-4
    )


def test_target_rank_above_delta_rank_pads_exactly_zero():
    """min(out, in) < target_rank is the branch that literally zero-pads."""
    state = make_state(out_f=4, in_f=20, rank=2, seed=4)
    merged = merge_states([state], [1.0], target_rank=6, alpha=ALPHA)
    a, b = merged['fc']['A'], merged['fc']['B']
    assert a.shape == (6, 20) and b.shape == (4, 6)
    assert torch.equal(a[4:], torch.zeros(2, 20))
    assert torch.equal(b[:, 4:], torch.zeros(4, 2))
    assert torch.allclose(
        lora_to_delta(merged, ALPHA)['fc'], lora_to_delta(state, ALPHA)['fc'], atol=1e-4
    )


def test_expansion_to_higher_rank_does_not_corrupt_the_update():
    """Expanding a rank-2 update to rank 6 keeps it; the extra components are noise."""
    state = make_state(rank=2, seed=4)
    merged = merge_states([state], [1.0], target_rank=6, alpha=ALPHA)
    a, b = merged['fc']['A'], merged['fc']['B']
    assert a.shape == (6, 20) and b.shape == (10, 6)

    original = lora_to_delta(state, ALPHA)['fc']
    assert torch.allclose(lora_to_delta(merged, ALPHA)['fc'], original, atol=1e-4)

    lead = (ALPHA / 6) * b[:, :2] @ a[:2]
    tail = (ALPHA / 6) * b[:, 2:] @ a[2:]
    assert tail.norm() < 1e-4 * lead.norm()


# --- optimality -------------------------------------------------------------

def test_merge_is_eckart_young_optimal():
    states = [make_state(rank=8, seed=s) for s in (1, 2, 3)]
    w = [0.2, 0.3, 0.5]
    dense = sum(wi * lora_to_delta(s, ALPHA)['fc'] for s, wi in zip(states, w))

    merged = merge_states(states, w, target_rank=3, alpha=ALPHA)
    got = lora_to_delta(merged, ALPHA)['fc']
    err = (got - dense).norm().item()

    u, s_vals, vh = torch.linalg.svd(dense, full_matrices=False)
    best = (u[:, :3] * s_vals[:3]) @ vh[:3, :]
    assert err <= (best - dense).norm().item() + 1e-4

    competitors = {
        'skip-top': (u[:, 1:4] * s_vals[1:4]) @ vh[1:4, :],
        'tail': (u[:, -3:] * s_vals[-3:]) @ vh[-3:, :],
    }
    g = torch.Generator().manual_seed(11)
    rand_b = torch.randn(10, 3, generator=g)
    competitors['random'] = rand_b @ torch.randn(3, 20, generator=g)
    for name, cand in competitors.items():
        assert err < (cand - dense).norm().item(), f"{name} beat the rank-3 merge"


# --- weights and degenerate input -------------------------------------------

def test_weights_are_normalised():
    states = [make_state(rank=4, seed=1), make_state(rank=4, seed=2)]
    a = merge_states(states, [2.0, 6.0], target_rank=8, alpha=ALPHA)
    b = merge_states(states, [0.25, 0.75], target_rank=8, alpha=ALPHA)
    assert torch.allclose(
        lora_to_delta(a, ALPHA)['fc'], lora_to_delta(b, ALPHA)['fc'], atol=1e-4
    )


def test_non_positive_weight_total_raises():
    states = [make_state(rank=4, seed=1), make_state(rank=4, seed=2)]
    with pytest.raises(ValueError, match="positive"):
        merge_states(states, [0.0, 0.0], target_rank=4, alpha=ALPHA)


def test_empty_input_raises():
    with pytest.raises(ValueError, match="at least one state"):
        merge_states([], [], target_rank=4, alpha=ALPHA)


def test_weight_count_mismatch_raises():
    states = [make_state(rank=4, seed=1), make_state(rank=4, seed=2)]
    with pytest.raises(ValueError, match="weights"):
        merge_states(states, [1.0], target_rank=4, alpha=ALPHA)
