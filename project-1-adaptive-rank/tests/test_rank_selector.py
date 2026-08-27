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


def test_floor_prevents_collapse_below_smallest_candidate():
    smallest = min(ALL_CANDIDATE_RANKS)
    for batch_size in BATCH_TO_MAX_RANK:
        for s in (0.0, 1e-6, 0.5):
            assert rank_equation(s, batch_size) >= smallest


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
