"""
Experiment 09: Scaling Alpha Policy Demo

Purpose:
    Demonstrate how to assign alpha values from rank assignments
    using the reusable scaling policy module.

Input:
    results/experiment_04_allocation_comparison/allocation_comparison.json

Output:
    results/experiment_09_scaling_policy_demo/scaling_policy_assignments.json
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.allocation.scaling_policy import (
    ScalingAlphaPolicy,
    EmpiricalScalingAlphaPolicy,
)


INPUT_PATH = "results/experiment_04_allocation_comparison/allocation_comparison.json"
OUTPUT_DIR = "results/experiment_09_scaling_policy_demo"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "scaling_policy_assignments.json")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_PATH, "r") as f:
        allocation_results = json.load(f)

    rank_assignments = allocation_results["Domain-Aware"]["rank_assignments"]

    fixed_scaling_policy = ScalingAlphaPolicy(target_scaling=1.0)
    empirical_policy = EmpiricalScalingAlphaPolicy()

    report = {
        "input_rank_assignments": rank_assignments,
        "fixed_scaling_policy": fixed_scaling_policy.recommend_for_clients(rank_assignments),
        "empirical_policy": empirical_policy.recommend_for_clients(rank_assignments),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("Scaling policy demo complete.")
    print(json.dumps(report, indent=2))
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
