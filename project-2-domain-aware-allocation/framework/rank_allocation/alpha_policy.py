import json
from collections import defaultdict


class AlphaPolicyAnalyzer:

    def __init__(self, json_path):
        self.json_path = json_path

        with open(json_path, "r") as f:
            self.data = json.load(f)

    def best_global_alpha(self):
        alpha_scores = defaultdict(list)

        for client_data in self.data.values():
            for alpha, acc in client_data["all_alpha_results"].items():
                alpha_scores[int(alpha)].append(acc)

        avg_scores = {}

        for alpha, scores in alpha_scores.items():
            avg_scores[alpha] = sum(scores) / len(scores)

        best_alpha = max(avg_scores, key=avg_scores.get)

        return {
            "best_alpha": best_alpha,
            "average_accuracy": avg_scores[best_alpha],
            "all_scores": avg_scores,
        }

    def best_alpha_per_rank(self):
        rank_scores = defaultdict(lambda: defaultdict(list))

        for client_data in self.data.values():
            rank = client_data["fixed_rank"]

            for alpha, acc in client_data["all_alpha_results"].items():
                rank_scores[rank][int(alpha)].append(acc)

        results = {}

        for rank, alpha_dict in rank_scores.items():

            avg_scores = {
                alpha: sum(scores) / len(scores)
                for alpha, scores in alpha_dict.items()
            }

            best_alpha = max(avg_scores, key=avg_scores.get)

            results[rank] = {
                "best_alpha": best_alpha,
                "average_accuracy": avg_scores[best_alpha],
            }

        return results
