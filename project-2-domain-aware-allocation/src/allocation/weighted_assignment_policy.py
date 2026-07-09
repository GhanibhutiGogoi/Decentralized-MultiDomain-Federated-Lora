"""
Weighted assignment policy for Project 2.

This module defines the Project 2 allocation-side formula for heterogeneous
federated LoRA training.

The policy maps client/domain signals into normalized assignment weights,
uses those weights to allocate a fixed total LoRA rank budget, and then
computes alpha/scaling values for each client.

This module does not replace Gabriel's adaptive rank work. Instead, it is
intended to connect with Gabriel's adaptive-rank output or agreed rank-output
format in the joint pipeline:

    Gabriel adaptive rank signal / rank output
        -> Project 2 weighted assignment
        -> alpha / scaling allocation
        -> matched-budget Experiment 04 evaluation

Pipeline:
    client signals -> score_i -> weight_i -> rank_i -> alpha_i -> scaling_i

The goal is to allocate a fixed total LoRA rank budget across clients using
normalized client/domain weights.
"""

from typing import Dict, List, Any


def nearest_allowed_rank(value: float, allowed_ranks: List[int]) -> int:
    """
    Map a continuous rank value to the nearest allowed LoRA rank.
    """
    return min(
        allowed_ranks,
        key=lambda rank: (abs(rank - value), rank),
    )


def compute_client_score(
    client_features: Dict[str, float],
    lambda_complexity: float = 1.0,
    lambda_entropy: float = 0.5,
    lambda_imbalance: float = 0.5,
) -> float:
    """
    Compute the weighted assignment score for one client.

    score_i = λ_c C_i + λ_e E_i + λ_b B_i

    where:
        C_i = complexity score
        E_i = entropy
        B_i = data imbalance

    Args:
        client_features:
            Dictionary containing client-level signals.
        lambda_complexity:
            Weight for complexity score.
        lambda_entropy:
            Weight for entropy.
        lambda_imbalance:
            Weight for data imbalance.

    Returns:
        Non-negative assignment score.
    """
    complexity = float(client_features.get("complexity_score", 0.0))
    entropy = float(client_features.get("entropy", 0.0))
    imbalance = float(client_features.get("data_imbalance", 0.0))

    score = (
        lambda_complexity * complexity
        + lambda_entropy * entropy
        + lambda_imbalance * imbalance
    )

    return max(score, 0.0)


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize client scores into assignment weights.

    w_i = score_i / sum_j score_j
    """
    total_score = sum(scores.values())

    if total_score <= 0:
        n_clients = len(scores)
        if n_clients == 0:
            return {}
        return {
            client_id: 1.0 / n_clients
            for client_id in scores
        }

    return {
        client_id: score / total_score
        for client_id, score in scores.items()
    }


def correct_rank_budget(
    rank_assignments: Dict[str, int],
    weights: Dict[str, float],
    target_budget: int,
    allowed_ranks: List[int],
) -> Dict[str, int]:
    """
    Correct total rank after nearest-rank rounding.

    Rounding continuous ranks to allowed LoRA ranks may make the total rank
    smaller or larger than the target budget. This function greedily adjusts
    ranks while respecting allowed ranks.

    If total rank is too high:
        reduce ranks for the lowest-weight clients first.

    If total rank is too low:
        increase ranks for the highest-weight clients first.
    """
    corrected = dict(rank_assignments)
    allowed_ranks = sorted(allowed_ranks)

    def current_total_rank() -> int:
        return sum(corrected.values())

    def lower_rank(rank: int):
        lower = [
            candidate
            for candidate in allowed_ranks
            if candidate < rank
        ]
        return max(lower) if lower else rank

    def higher_rank(rank: int):
        higher = [
            candidate
            for candidate in allowed_ranks
            if candidate > rank
        ]
        return min(higher) if higher else rank

    # Reduce rank if above budget
    while current_total_rank() > target_budget:
        changed = False

        for client_id, _ in sorted(
            weights.items(),
            key=lambda item: item[1],
        ):
            old_rank = corrected[client_id]
            new_rank = lower_rank(old_rank)

            if new_rank < old_rank:
                corrected[client_id] = new_rank
                changed = True
                break

        if not changed:
            break

    # Increase rank if below budget
    while current_total_rank() < target_budget:
        changed = False

        for client_id, _ in sorted(
            weights.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            old_rank = corrected[client_id]
            new_rank = higher_rank(old_rank)

            if new_rank > old_rank:
                if current_total_rank() - old_rank + new_rank <= target_budget:
                    corrected[client_id] = new_rank
                    changed = True
                    break

        if not changed:
            break

    return corrected


def weighted_assignment(
    client_features: Dict[str, Dict[str, Any]],
    total_rank_budget: int,
    allowed_ranks: List[int] = None,
    min_rank: int = 4,
    alpha: int = 64,
    lambda_complexity: float = 1.0,
    lambda_entropy: float = 0.5,
    lambda_imbalance: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute weighted LoRA assignment under a fixed total rank budget.

    Formula:

        score_i = λ_c C_i + λ_e E_i + λ_b B_i

        w_i = score_i / sum_j score_j

        raw_rank_i = r_min + w_i * (B_total - N * r_min)

        r_i = nearest_allowed_rank(raw_rank_i)

        scaling_i = alpha_i / r_i

    Args:
        client_features:
            Dictionary mapping client_id to client/domain signals.
        total_rank_budget:
            Fixed total rank budget to allocate across clients.
        allowed_ranks:
            Valid LoRA ranks.
        min_rank:
            Minimum rank assigned to every client before weighted allocation.
        alpha:
            Global LoRA alpha used in this first policy version.
        lambda_complexity:
            Weight for complexity score.
        lambda_entropy:
            Weight for entropy.
        lambda_imbalance:
            Weight for data imbalance.

    Returns:
        Dictionary containing per-client allocations and metadata.
    """
    if allowed_ranks is None:
        allowed_ranks = [4, 8, 16, 32, 64]

    if not client_features:
        raise ValueError("client_features must not be empty.")

    n_clients = len(client_features)

    minimum_required_budget = n_clients * min_rank

    if total_rank_budget < minimum_required_budget:
        raise ValueError(
            "total_rank_budget must be at least n_clients * min_rank."
        )

    scores = {
        client_id: compute_client_score(
            features,
            lambda_complexity=lambda_complexity,
            lambda_entropy=lambda_entropy,
            lambda_imbalance=lambda_imbalance,
        )
        for client_id, features in client_features.items()
    }

    weights = normalize_scores(scores)

    remaining_budget = total_rank_budget - minimum_required_budget

    raw_ranks = {
        client_id: min_rank + weights[client_id] * remaining_budget
        for client_id in client_features
    }

    initial_rank_assignments = {
        client_id: nearest_allowed_rank(
            raw_rank,
            allowed_ranks,
        )
        for client_id, raw_rank in raw_ranks.items()
    }

    rank_assignments = correct_rank_budget(
        rank_assignments=initial_rank_assignments,
        weights=weights,
        target_budget=total_rank_budget,
        allowed_ranks=allowed_ranks,
    )

    allocations = {}

    for client_id in client_features:
        rank = rank_assignments[client_id]

        allocations[client_id] = {
            "score": scores[client_id],
            "weight": weights[client_id],
            "raw_rank": raw_ranks[client_id],
            "rank": rank,
            "alpha": alpha,
            "scaling": alpha / rank,
            "features": client_features[client_id],
        }

    actual_total_rank = sum(
        allocation["rank"]
        for allocation in allocations.values()
    )

    return {
        "formula": {
            "score": "score_i = lambda_complexity * complexity_i + lambda_entropy * entropy_i + lambda_imbalance * imbalance_i",
            "weight": "w_i = score_i / sum_j score_j",
            "raw_rank": "raw_rank_i = r_min + w_i * (B_total - N * r_min)",
            "rank": "r_i = nearest_allowed_rank(raw_rank_i)",
            "scaling": "scaling_i = alpha_i / r_i",
        },
        "hyperparameters": {
            "total_rank_budget": total_rank_budget,
            "actual_total_rank": actual_total_rank,
            "budget_error": actual_total_rank - total_rank_budget,
            "allowed_ranks": allowed_ranks,
            "min_rank": min_rank,
            "alpha": alpha,
            "lambda_complexity": lambda_complexity,
            "lambda_entropy": lambda_entropy,
            "lambda_imbalance": lambda_imbalance,
        },
        "allocations": allocations,
    }
