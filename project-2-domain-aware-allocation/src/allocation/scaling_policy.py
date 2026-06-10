"""
Scaling-based Alpha Policy for LoRA.

LoRA uses effective scaling:

    scaling = alpha / rank

This module provides reusable policies for selecting alpha from rank.
"""


class ScalingAlphaPolicy:
    def __init__(
        self,
        target_scaling=1.0,
        candidate_alphas=None,
        min_alpha=4,
        max_alpha=128,
    ):
        self.target_scaling = target_scaling
        self.candidate_alphas = candidate_alphas or [4, 8, 16, 32, 64, 128]
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    def recommend_alpha(self, rank):
        """
        Recommend alpha using:

            alpha = rank * target_scaling

        Then snap to the nearest candidate alpha.
        """
        raw_alpha = rank * self.target_scaling

        raw_alpha = max(self.min_alpha, min(self.max_alpha, raw_alpha))

        best_alpha = min(
            self.candidate_alphas,
            key=lambda a: abs(a - raw_alpha),
        )

        return int(best_alpha)

    def recommend_for_clients(self, rank_assignments):
        """
        Args:
            rank_assignments:
                dict like {"0": 32, "1": 64}

        Returns:
            dict like {"0": {"rank": 32, "alpha": 32, "scaling": 1.0}}
        """
        result = {}

        for client_id, rank in rank_assignments.items():
            rank = int(rank)
            alpha = self.recommend_alpha(rank)

            result[str(client_id)] = {
                "rank": rank,
                "alpha": alpha,
                "scaling": alpha / rank,
            }

        return result


class EmpiricalScalingAlphaPolicy:
    """
    Empirical policy based on Experiment 08.

    Experiment 08 showed that higher ranks tend to require smaller scaling.
    This policy uses simple rank-specific scaling choices.
    """

    def __init__(self):
        self.rank_to_alpha = {
            4: 32,
            8: 32,
            16: 64,
            32: 64,
            64: 64,
        }

    def recommend_alpha(self, rank):
        rank = int(rank)

        if rank in self.rank_to_alpha:
            return self.rank_to_alpha[rank]

        # fallback
        return 64

    def recommend_for_clients(self, rank_assignments):
        result = {}

        for client_id, rank in rank_assignments.items():
            rank = int(rank)
            alpha = self.recommend_alpha(rank)

            result[str(client_id)] = {
                "rank": rank,
                "alpha": alpha,
                "scaling": alpha / rank,
            }

        return result
