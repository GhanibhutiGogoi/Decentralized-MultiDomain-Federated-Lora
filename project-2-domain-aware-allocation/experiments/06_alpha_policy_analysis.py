"""
Experiment 06: Alpha Policy Analysis

Purpose:
    Analyze Experiment 05 alpha-search results and derive
    simple alpha allocation policies.

Input:
    results/experiment_05_alpha_search/best_alpha_per_client.json

Output:
    results/experiment_06_alpha_policy_analysis/alpha_policy_report.json
"""

import os
import json
from collections import defaultdict


INPUT_FILE = (
    "results/experiment_05_alpha_search/"
    "best_alpha_per_client.json"
)

OUTPUT_DIR = "results/experiment_06_alpha_policy_analysis"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "alpha_policy_report.json"
)


def load_results():
    with open(INPUT_FILE, "r") as f:
        return json.load(f)


def compute_global_alpha(data):
    alpha_scores = defaultdict(list)

    for client in data.values():
        for alpha, acc in client["all_alpha_results"].items():
            alpha_scores[int(alpha)].append(acc)

    avg_scores = {
        alpha: sum(scores) / len(scores)
        for alpha, scores in alpha_scores.items()
    }

    best_alpha = max(avg_scores, key=avg_scores.get)

    return {
        "best_alpha": best_alpha,
        "average_accuracy": avg_scores[best_alpha],
        "all_average_scores": dict(sorted(avg_scores.items()))
    }


def compute_alpha_distribution(data):
    distribution = defaultdict(int)

    for client in data.values():
        distribution[client["best_alpha"]] += 1

    return dict(sorted(distribution.items()))


def compute_rank_specific_alpha(data):
    rank_scores = defaultdict(lambda: defaultdict(list))

    for client in data.values():
        rank = client["fixed_rank"]

        for alpha, acc in client["all_alpha_results"].items():
            rank_scores[rank][int(alpha)].append(acc)

    result = {}

    for rank, alpha_dict in rank_scores.items():

        avg_scores = {
            alpha: sum(scores) / len(scores)
            for alpha, scores in alpha_dict.items()
        }

        best_alpha = max(avg_scores, key=avg_scores.get)

        result[rank] = {
            "best_alpha": best_alpha,
            "average_accuracy": avg_scores[best_alpha],
            "all_average_scores": dict(sorted(avg_scores.items()))
        }

    return dict(sorted(result.items()))


def compute_tolerance_policy(data, tolerance=0.005):
    """
    Smallest alpha within tolerance of oracle accuracy.
    """

    result = {}

    for client_id, client in data.items():

        best_acc = client["best_accuracy"]

        candidates = []

        for alpha, acc in client["all_alpha_results"].items():

            alpha = int(alpha)

            if acc >= best_acc - tolerance:
                candidates.append((alpha, acc))

        chosen_alpha, chosen_acc = min(
            candidates,
            key=lambda x: x[0]
        )

        result[client_id] = {
            "fixed_rank": client["fixed_rank"],
            "oracle_best_alpha": client["best_alpha"],
            "oracle_best_accuracy": best_acc,
            "policy_alpha": chosen_alpha,
            "policy_accuracy": chosen_acc,
            "accuracy_drop": best_acc - chosen_acc
        }

    return result


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_results()

    report = {
        "best_global_alpha":
            compute_global_alpha(data),

        "best_alpha_distribution":
            compute_alpha_distribution(data),

        "best_alpha_per_rank":
            compute_rank_specific_alpha(data),

        "smallest_alpha_within_0_005_tolerance":
            compute_tolerance_policy(
                data,
                tolerance=0.005
            )
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print("EXPERIMENT 06 COMPLETE")
    print("=" * 60)

    print("\nBest Global Alpha:")
    print(report["best_global_alpha"])

    print("\nAlpha Distribution:")
    print(report["best_alpha_distribution"])

    print("\nBest Alpha Per Rank:")
    print(report["best_alpha_per_rank"])

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
