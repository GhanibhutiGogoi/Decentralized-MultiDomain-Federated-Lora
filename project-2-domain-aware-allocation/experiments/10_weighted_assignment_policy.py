
"""
Experiment 10: Weighted Assignment Policy

Purpose:
    Define and test the weighted assignment formula for Project 2.

    This experiment maps client/domain signals into normalized assignment
    weights and uses those weights to distribute a fixed LoRA rank budget.

    Pipeline:
        client signals -> score_i -> weight_i -> rank_i -> alpha_i -> scaling_i

Outputs:
    - results/experiment_10_weighted_assignment_policy/weighted_assignment_results.json

Note:
    This is a formula / policy validation experiment, not a final performance
    comparison.
"""

import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.allocation.weighted_assignment_policy import weighted_assignment


RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "experiment_10_weighted_assignment_policy"
)

OUTPUT_FILE = RESULTS_DIR / "weighted_assignment_results.json"


def load_client_features():
    """
    Load client features for weighted assignment.

    This first version uses the client table from Experiment 08 scaling analysis,
    because it already contains the signals needed for the formula:
        - complexity_score
        - entropy
        - data_imbalance

    Returns:
        Dictionary mapping client_id string -> feature dictionary.
    """
    scaling_path = (
        PROJECT_ROOT
        / "results"
        / "experiment_08_scaling_analysis"
        / "scaling_analysis.json"
    )

    if not scaling_path.exists():
        raise FileNotFoundError(
            f"Could not find required input file: {scaling_path}"
        )

    with open(scaling_path, "r") as f:
        scaling_data = json.load(f)

    client_features = {}

    for row in scaling_data["client_table"]:
        client_id = f"client_{row['client_id']}"

        client_features[client_id] = {
            "client_id": row["client_id"],
            "domain_id": row["domain_id"],
            "domain_name": row["domain_name"],
            "complexity_score": row["complexity_score"],
            "entropy": row["entropy"],
            "data_imbalance": row["data_imbalance"],
        }

    return client_features


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    client_features = load_client_features()

    # Matched-budget reference:
    # Uniform rank 16 over 15 clients gives total_rank_budget = 240.
    total_rank_budget = 240

    results = weighted_assignment(
        client_features=client_features,
        total_rank_budget=total_rank_budget,
        allowed_ranks=[4, 8, 16, 32, 64],
        min_rank=4,
        alpha=64,
        lambda_complexity=1.0,
        lambda_entropy=0.5,
        lambda_imbalance=0.5,
    )

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 80)
    print("Experiment 10: Weighted Assignment Policy")
    print("=" * 80)
    print(f"Output saved to: {OUTPUT_FILE}")
    print()
    print("Target total rank:", results["hyperparameters"]["total_rank_budget"])
    print("Actual total rank:", results["hyperparameters"]["actual_total_rank"])
    print("Budget error:", results["hyperparameters"]["budget_error"])
    print()
    print("Client allocations:")

    for client_id, allocation in results["allocations"].items():
        print(
            client_id,
            "| weight:",
            round(allocation["weight"], 4),
            "| raw_rank:",
            round(allocation["raw_rank"], 2),
            "| rank:",
            allocation["rank"],
            "| alpha:",
            allocation["alpha"],
            "| scaling:",
            round(allocation["scaling"], 2),
        )

    print("=" * 80)
    print("Experiment 10 complete!")


if __name__ == "__main__":
    main()
