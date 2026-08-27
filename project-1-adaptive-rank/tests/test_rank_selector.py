"""Regression tests for the capability-aware rank equation (audit defect D1)."""

import pytest

from config import ALL_CANDIDATE_RANKS, BATCH_TO_MAX_RANK
from rank_allocation.rank_selector import (
    GAMMA,
    capability_fraction,
    rank_equation,
)

TOP_BATCH = max(BATCH_TO_MAX_RANK)
STABLE_RANKS = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 50.0)


def test_top_tier_rank_varies_with_stable_rank():
    """D1: at c_i = 1 the floor used to equal R^max, so s(G) was discarded."""
    assert capability_fraction(TOP_BATCH) == 1.0
    chosen = {rank_equation(s, TOP_BATCH) for s in STABLE_RANKS}
    assert len(chosen) > 1, f"rank is constant at the top tier: {chosen}"


def test_hardware_ceiling_is_never_exceeded():
    for batch_size, max_rank in BATCH_TO_MAX_RANK.items():
        for s in STABLE_RANKS + (10_000.0,):
            assert rank_equation(s, batch_size) <= max_rank


@pytest.mark.parametrize("gamma", (0.25, 0.5, 0.875, 0.9, 1.0))
def test_capability_floor_is_a_genuine_lower_bound(gamma):
    """The chosen rank must never fall below the capability floor itself.

    Asserting only `>= min(ALL_CANDIDATE_RANKS)` would be vacuous: the function
    always returns an element of the candidate menu, so that holds for any
    implementation, any gamma, and the pre-fix code. The binding claim is
    against gamma * c_i * R_max, and it needs the upward snap in rank_equation
    because _nearest_candidate breaks ties downward -- at gamma=0.875 the floor
    is 14.0 and the nearest candidate is 12.
    """
    for batch_size, max_rank in BATCH_TO_MAX_RANK.items():
        candidates = [r for r in ALL_CANDIDATE_RANKS if r <= max_rank]
        floor = max(candidates[0], gamma * capability_fraction(batch_size) * max_rank)
        reachable = [r for r in candidates if r >= floor]
        if not reachable:
            continue  # floor above the whole menu: the ceiling wins, tested elsewhere
        for s in (0.0, 1e-6, 0.5, 1.0):
            assert rank_equation(s, batch_size, gamma=gamma) >= floor


def test_non_finite_stable_rank_falls_back_to_the_floor():
    """A diverged gradient probe must not propagate NaN into the allocation."""
    for batch_size, max_rank in BATCH_TO_MAX_RANK.items():
        candidates = [r for r in ALL_CANDIDATE_RANKS if r <= max_rank]
        floor = max(candidates[0], GAMMA * capability_fraction(batch_size) * max_rank)
        expected = rank_equation(0.0, batch_size)
        for bad in (float("nan"), float("inf"), float("-inf")):
            chosen = rank_equation(bad, batch_size)
            assert chosen in candidates
            assert chosen >= floor
            # equality, not just >= floor: without the guard +inf would sail
            # past the floor to the hardware ceiling instead of falling back.
            assert chosen == expected


def test_capability_floor_still_binds_at_top_tier():
    """The soft prior must still lift a tiny demand above the smallest candidate."""
    assert rank_equation(0.1, TOP_BATCH) > min(ALL_CANDIDATE_RANKS)


def test_gamma_one_reproduces_the_old_jammed_behaviour():
    """Regression guard: gamma=1 is the pre-fix formula, constant at the top tier."""
    chosen = {rank_equation(s, TOP_BATCH, gamma=1.0) for s in STABLE_RANKS}
    assert chosen == {BATCH_TO_MAX_RANK[TOP_BATCH]}


def test_gamma_default_is_a_soft_prior():
    assert 0.0 < GAMMA < 1.0


@pytest.mark.parametrize("batch_size", sorted(BATCH_TO_MAX_RANK))
def test_returned_rank_is_always_a_candidate(batch_size):
    for s in STABLE_RANKS:
        assert rank_equation(s, batch_size) in ALL_CANDIDATE_RANKS


def test_shipped_gamma_allocation_is_still_capability_dominated():
    """Pins a known limitation so a future recalibration is visible.

    s(G) is bounded by the probe rank (a stable rank cannot exceed the matrix
    rank), and measured s(G) on the real project-1 models sits in roughly
    [1.1, 4.2]. At GAMMA = 0.5 the top-tier floor is 8, so over that whole
    observed range the allocation does not move: it is (2, 2, 8), down from the
    pre-fix (2, 4, 16) but still a function of batch size alone.

    This test asserts the limitation, not the desired behaviour. If someone
    recalibrates GAMMA or changes the demand term, it should fail and be
    updated deliberately.
    """
    observed = (1.1, 1.4, 1.9, 2.8, 3.4, 4.2)

    # The top tier is pinned at its floor of 8 across the entire observed range:
    # gamma * c * R_max = 0.5 * 1 * 16 = 8, and s(G) never gets near it.
    assert {rank_equation(s, 256) for s in observed} == {8}
    assert rank_equation(9.9, 256) == 8      # still pinned just below the knee
    assert rank_equation(12.0, 256) == 12    # demand binds once it clears 8

    # The two lower tiers do respond within the observed range, but only up to
    # their hardware ceilings of 4 and 8 -- so their headroom is small.
    assert rank_equation(1.1, 16) == 2 and rank_equation(4.2, 16) == 4
    assert rank_equation(1.1, 64) == 2 and rank_equation(4.2, 64) == 4
