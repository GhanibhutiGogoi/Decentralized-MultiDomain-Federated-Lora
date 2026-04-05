"""
LoRA Rank Allocation Strategies for Heterogeneous Federated Learning.

Provides multiple strategies for assigning LoRA ranks to clients:
    - Uniform: all clients get the same rank
    - Random: each client gets a random rank
    - Domain-aware: rank proportional to domain complexity score
    - Oracle: best rank found by exhaustive search
"""

import numpy as np

CANDIDATE_RANKS = [4, 8, 16, 32, 64]


def snap_to_nearest_rank(rank, candidates=None):
    """Snap a continuous rank value to the nearest candidate rank."""
    if candidates is None:
        candidates = CANDIDATE_RANKS
    return min(candidates, key=lambda x: abs(x - rank))


class RankAllocator:
    """Allocates LoRA ranks to federated clients using different strategies."""

    @staticmethod
    def uniform(n_clients, rank=16):
        """
        All clients get the same rank.

        Args:
            n_clients: number of clients
            rank: the uniform rank to assign

        Returns:
            dict mapping client_id -> rank
        """
        return {i: rank for i in range(n_clients)}

    @staticmethod
    def random(n_clients, ranks=None, seed=42):
        """
        Each client gets a random rank from candidate ranks.

        Args:
            n_clients: number of clients
            ranks: list of candidate ranks
            seed: random seed for reproducibility

        Returns:
            dict mapping client_id -> rank
        """
        if ranks is None:
            ranks = CANDIDATE_RANKS
        rng = np.random.default_rng(seed)
        return {i: int(rng.choice(ranks)) for i in range(n_clients)}

    @staticmethod
    def domain_aware(complexity_scores, min_rank=4, max_rank=64, base_rank=16):
        """
        Allocate rank proportional to domain complexity score.

        Formula: rank = snap(base_rank * (1 + 3 * complexity_score))
        - complexity_score = 0 -> rank = base_rank (16)
        - complexity_score = 0.5 -> rank = 40 -> snaps to 32
        - complexity_score = 1.0 -> rank = 64

        Args:
            complexity_scores: dict mapping client_id -> float in [0,1]
            min_rank: minimum allowed rank
            max_rank: maximum allowed rank
            base_rank: starting rank before complexity scaling

        Returns:
            dict mapping client_id -> rank
        """
        allocations = {}
        for client_id, score in complexity_scores.items():
            raw_rank = base_rank * (1 + 3 * score)
            rank = snap_to_nearest_rank(raw_rank)
            rank = max(min_rank, min(max_rank, rank))
            allocations[client_id] = rank
        return allocations

    @staticmethod
    def oracle(oracle_results):
        """
        Use the best rank found by exhaustive grid search.

        Args:
            oracle_results: dict mapping client_id -> {rank -> accuracy}

        Returns:
            dict mapping client_id -> best_rank
        """
        allocations = {}
        for client_id, rank_accuracies in oracle_results.items():
            best_rank = max(rank_accuracies, key=rank_accuracies.get)
            allocations[int(client_id)] = int(best_rank)
        return allocations

    @staticmethod
    def summarize(allocations, complexity_scores=None):
        """Print a summary of rank allocations."""
        print("\n" + "=" * 50)
        print("Rank Allocation Summary")
        print("=" * 50)
        for cid in sorted(allocations.keys()):
            line = f"  Client {cid:2d} -> Rank {allocations[cid]:3d}"
            if complexity_scores and cid in complexity_scores:
                line += f"  (complexity: {complexity_scores[cid]:.3f})"
            print(line)
        ranks = list(allocations.values())
        print(f"\n  Mean rank: {np.mean(ranks):.1f}")
        print(f"  Rank distribution: {dict(zip(*np.unique(ranks, return_counts=True)))}")
        print("=" * 50 + "\n")
