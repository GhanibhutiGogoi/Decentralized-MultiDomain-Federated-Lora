"""Capability-aware adaptive LoRA rank selection."""

import math

import numpy as np
import torch

from config import ALL_CANDIDATE_RANKS, BATCH_TO_MAX_RANK
from Federated.client import set_lora_only_trainable

# Weight on the capability prior in the rank floor. gamma < 1 keeps the
# capability term a soft prior so the measured demand term can bind; gamma = 1
# reproduces the pre-fix behaviour, where the floor equalled the hardware
# ceiling at the top tier and the stable-rank measurement was discarded.
#
# 0.5 is the midpoint of the usable interval (0, 1) and is deliberately not
# tuned: it reserves half of each client's hardware budget as a floor and
# leaves the other half for the gradient measurement to claim. It is a
# published constant of the allocation rule, so it is recorded alongside the
# equation in the experiment artifact rather than left implicit. Note the
# consequence at the middle tier of the shipped BATCH_TO_MAX_RANK: the floor
# gamma * 0.5 * 8 = 2 coincides with the smallest candidate rank, so there the
# prior is inert by construction and demand alone decides.
#
# CALIBRATION IS UNRESOLVED. Fixing the floor == ceiling defect makes the demand
# term *reachable*, but it does not by itself make the shipped allocation
# responsive. Two facts, measured on the real project-1 models:
#
#   1. s(G) is bounded above by the probe's own rank. The probe is built at
#      BATCH_TO_MAX_RANK[batch_size], and the stable rank of an [r, in] or
#      [out, r] gradient cannot exceed r, so s(G) <= R_i^max always and the
#      min(s(G), R_i^max) cap below is mathematically redundant. It is kept
#      because it documents the intended semantics and costs nothing.
#   2. Observed s(G) on MLP and CNN backbones sits in roughly [1.1, 4.2] both at
#      initialisation and after several epochs, while the top-tier floor at
#      gamma = 0.5 is 8. The top tier therefore stays pinned at 8 across that
#      whole range; it does not move until s(G) passes 10.1, because the rank
#      menu quantises and _nearest_candidate rounds toward 8 until s(G) clears
#      the 8/12 midpoint. The two lower tiers do respond, but their ceilings
#      are 4 and 8, so the headroom is small.
#
# Choosing a smaller gamma from these numbers alone would be guesswork: the
# measurements above are from short synthetic probes, not the full benchmark
# battery. Settling it needs a rank-vs-accuracy sweep on the real tasks. Until
# then the honest description of this rule is capability-dominated, and
# test_shipped_gamma_allocation_is_still_capability_dominated pins that so a
# future recalibration is visible rather than silent.
GAMMA = 0.5


class AdaptiveRankController:
    """Stateful adaptive-rank policy with EMA, hysteresis and patience.

    ``rank_equation`` is intentionally stateless and useful for one-off probes,
    but a federated client should not oscillate between adjacent ranks because
    of noisy mini-batches.  This controller smooths the demand signal and only
    changes rank after ``patience`` consecutive requests in the same direction.
    A residual ratio (if supplied) raises demand when the current rank is
    losing update energy; it is clipped to keep a bad probe from exploding the
    allocation.  All decisions remain bounded by the client's capability menu.
    """

    def __init__(self, batch_size, initial_rank=None, candidates=None, gamma=GAMMA,
                 ema_decay=0.7, up_margin=0.10, down_margin=0.15, patience=2,
                 residual_weight=0.5, warmup_rounds=0, min_rank=None,
                 quality_drop_tolerance=0.05):
        self.batch_size = batch_size
        capability_max = BATCH_TO_MAX_RANK.get(batch_size, min(ALL_CANDIDATE_RANKS))
        # An explicit menu can narrow the choices but may not bypass the
        # hardware ceiling advertised by BATCH_TO_MAX_RANK.
        menu = candidates if candidates is not None else ALL_CANDIDATE_RANKS
        self.candidates = tuple(sorted({int(r) for r in menu if int(r) <= capability_max}))
        if not self.candidates:
            self.candidates = (min(ALL_CANDIDATE_RANKS),)
        self.gamma = float(gamma)
        self.ema_decay = float(ema_decay)
        self.up_margin = float(up_margin)
        self.down_margin = float(down_margin)
        self.patience = max(1, int(patience))
        self.residual_weight = float(residual_weight)
        self.warmup_rounds = max(0, int(warmup_rounds))
        self.rounds_seen = 0
        if min_rank is None:
            # Generic controller instances retain the historical menu floor;
            # experiment drivers can opt into a conservative capability floor.
            min_rank = (max(self.candidates[0], int(math.ceil(0.5 * capability_max)))
                        if initial_rank == "max" else self.candidates[0])
        self.min_rank = _ceil_candidate(float(min_rank), self.candidates)
        self.quality_drop_tolerance = max(0.0, float(quality_drop_tolerance))
        self.quality_ema = None
        self._quality_alarm = False
        if initial_rank == "max":
            self.rank = self.max_rank
        else:
            self.rank = int(initial_rank) if initial_rank in self.candidates else self.candidates[0]
            self.rank = max(self.rank, self.min_rank)
        self.ema_demand = None
        self._direction = 0
        self._streak = 0
        # Last decision details are intentionally public read-only-by-convention
        # attributes.  Experiment drivers can persist these alongside rank
        # histories so an allocation curve remains auditable after the run.
        self.last_demand = None
        self.last_stable_rank = None
        self.last_residual_ratio = None
        self.last_target_rank = self.rank
        self.last_direction = 0
        self.last_changed = False

    def observe_quality(self, quality):
        """Record post-training quality for a conservative next-round guard.

        A sudden quality drop after a rank decrease arms an increase to the
        capability ceiling on the next update.  The guard is deliberately
        relative, since absolute losses differ substantially by task.
        """
        try:
            q = float(quality)
        except (TypeError, ValueError):
            return
        if not math.isfinite(q) or q <= 0:
            return
        if self.quality_ema is not None and q < self.quality_ema * (1.0 - self.quality_drop_tolerance):
            self._quality_alarm = True
        self.quality_ema = q if self.quality_ema is None else 0.8 * self.quality_ema + 0.2 * q

    @property
    def max_rank(self):
        return self.candidates[-1]

    def update(self, stable_rank, residual_ratio=None):
        """Update state and return the rank to use on the next local step."""
        try:
            demand = float(stable_rank)
        except (TypeError, ValueError):
            demand = 0.0
        if not math.isfinite(demand):
            demand = 0.0
        self.rounds_seen += 1
        self.last_stable_rank = demand
        self.last_residual_ratio = None
        if residual_ratio is not None:
            try:
                rr = float(residual_ratio)
            except (TypeError, ValueError):
                rr = 0.0
            if math.isfinite(rr):
                self.last_residual_ratio = rr
                demand *= 1.0 + self.residual_weight * min(max(rr, 0.0), 1.0)
        demand = min(max(demand, 0.0), float(self.max_rank))
        self.last_demand = demand
        self.ema_demand = demand if self.ema_demand is None else (
            self.ema_decay * self.ema_demand + (1.0 - self.ema_decay) * demand
        )
        target = rank_equation(self.ema_demand, self.batch_size, gamma=self.gamma)
        force_restore = self.rounds_seen <= self.warmup_rounds or self._quality_alarm
        restored = False
        if force_restore:
            # Begin from the richest feasible adapter so the global model can
            # establish a useful update.  If quality regresses after a
            # reduction, restore the ceiling before considering reductions.
            target = self.max_rank
            self._quality_alarm = False
            if self.rank < self.max_rank:
                self.rank = self.max_rank
                self._direction, self._streak = 0, 0
                restored = True
        # Restrict to an explicitly supplied menu as well as the global menu.
        target = _nearest_candidate(target, self.candidates)
        self.last_target_rank = target
        direction = 1 if target > self.rank and self.ema_demand >= self.rank * (1.0 + self.up_margin) else \
            -1 if target < self.rank and self.ema_demand <= self.rank * (1.0 - self.down_margin) else 0
        self.last_direction = direction
        self.last_changed = restored
        if direction == 0:
            self._direction, self._streak = 0, 0
            return self.rank
        if direction == self._direction:
            self._streak += 1
        else:
            self._direction, self._streak = direction, 1
        if self._streak >= self.patience:
            idx = self.candidates.index(self.rank) + direction
            idx = min(max(idx, 0), len(self.candidates) - 1)
            if self.candidates[idx] < self.min_rank:
                idx = self.candidates.index(self.min_rank)
            old_rank = self.rank
            self.rank = self.candidates[idx]
            self.last_changed = self.rank != old_rank
            self._streak = 0
        return self.rank

    def diagnostics(self):
        """Return a serialisable snapshot of the latest controller decision."""
        return {
            "rank": int(self.rank),
            "demand": None if self.last_demand is None else float(self.last_demand),
            "stable_rank": None if self.last_stable_rank is None else float(self.last_stable_rank),
            "residual_ratio": self.last_residual_ratio,
            # Tail mass is a ΔW compression diagnostic and is unavailable to
            # the gradient-only controller; retain an explicit null field so
            # downstream artifact schemas stay stable across policies.
            "tail_mass": None,
            "ema_demand": None if self.ema_demand is None else float(self.ema_demand),
            "target_rank": int(self.last_target_rank),
            "direction": int(self.last_direction),
            "streak": int(self._streak),
            "changed": bool(self.last_changed),
            "max_rank": int(self.max_rank),
            "gamma": float(self.gamma),
            "warmup_rounds": int(self.warmup_rounds),
            "rounds_seen": int(self.rounds_seen),
            "min_rank": int(self.min_rank),
            "quality_ema": None if self.quality_ema is None else float(self.quality_ema),
            "quality_alarm": bool(self._quality_alarm),
        }


def _nearest_candidate(rank, candidates):
    return min(candidates, key=lambda r: (abs(r - rank), r))


def _ceil_candidate(rank, candidates):
    """Choose the smallest menu entry greater than or equal to ``rank``."""
    above = [r for r in candidates if r >= rank]
    return min(above) if above else max(candidates)


def capability_fraction(batch_size):
    """Map client batch-size capability to [0, 1]."""
    batch_sizes = sorted(BATCH_TO_MAX_RANK)
    if batch_size not in BATCH_TO_MAX_RANK:
        return 0.0
    if len(batch_sizes) == 1:
        return 1.0
    return batch_sizes.index(batch_size) / (len(batch_sizes) - 1)


def rank_equation(stable_rank, batch_size, gamma=GAMMA):
    r"""
    Closed-form adaptive rank rule.

        s(G) = ||G||_F^2 / ||G||_2^2
        c_i  = (index(batch_i) / (num_capabilities - 1))
        r_i  = round_to_candidate(
                 max(2, gamma * c_i * R_i^max, min(s(G), R_i^max))
               )

    The stable rank estimates update complexity from the gradient geometry.
    The capability term is a *soft prior* (weight gamma < 1) that keeps stronger
    clients from being under-allocated without pinning them to their ceiling;
    the ceiling prevents weaker clients from exceeding their budget.

    With gamma = 1 the floor equals R_i^max whenever c_i = 1, and since the
    demand term is capped at R_i^max the outer max always returns the floor --
    the measured stable rank is computed and then discarded. gamma < 1 leaves
    room between the floor and the ceiling for s(G) to bind.
    """
    max_rank = BATCH_TO_MAX_RANK.get(batch_size, min(ALL_CANDIDATE_RANKS))
    candidates = [r for r in ALL_CANDIDATE_RANKS if r <= max_rank]
    if not candidates:
        return min(ALL_CANDIDATE_RANKS)

    # A non-finite measurement means the gradient probe diverged. Fall back to
    # the capability floor deliberately rather than letting NaN propagate
    # silently through max/min, where it would collapse to the floor anyway but
    # by accident.
    demand = float(stable_rank)
    if not math.isfinite(demand):
        demand = 0.0

    floor = max(candidates[0], gamma * capability_fraction(batch_size) * max_rank)
    raw_rank = max(floor, min(demand, float(max_rank)))

    chosen = _nearest_candidate(raw_rank, candidates)
    # _nearest_candidate breaks ties downward, which can land below the floor
    # when the floor sits between two candidates. Snap back up so the floor is
    # a genuine lower bound for any gamma, not just the shipped one.
    if chosen < floor:
        above = [r for r in candidates if r >= floor]
        if above:
            chosen = min(above)
    return chosen


def estimate_gradient_stable_rank(model, loader, loss_fn, num_batches=3):
    """Estimate median gradient stable rank from trainable 2-D parameters."""
    model.train()
    set_lora_only_trainable(model)
    stable_ranks = []
    device = next(model.parameters()).device

    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= num_batches:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()

        for param in model.parameters():
            if param.requires_grad and param.grad is not None and param.grad.dim() == 2:
                grad = param.grad.float()
                frob_sq = torch.sum(grad * grad).item()
                spectral_sq = torch.linalg.matrix_norm(grad, ord=2).item() ** 2
                if spectral_sq > 1e-12:
                    stable_ranks.append(frob_sq / spectral_sq)

    return float(np.median(stable_ranks)) if stable_ranks else 1.0


def estimate_optimal_rank(model, loader, loss_fn, batch_size, num_batches=3):
    """Select adaptive rank using the closed-form rank equation."""
    stable_rank = estimate_gradient_stable_rank(model, loader, loss_fn, num_batches)
    return rank_equation(stable_rank, batch_size)
