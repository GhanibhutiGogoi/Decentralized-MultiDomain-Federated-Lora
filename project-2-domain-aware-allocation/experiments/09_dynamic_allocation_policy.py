"""
Experiment 09: Dynamic Allocation Policy

Purpose:
    Connect Gabriel's adaptive rank-selection output with Project 2
    alpha/scaling allocation policy.

Input:
    Gabriel's rank-selection output:
        client_i -> selected rank r_i

Output:
    results/experiment_09_dynamic_allocation_policy/
        dynamic_allocation_results.json

This first version uses a global alpha value of 64 based on
Experiment 06 alpha-policy analysis.
"""

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.allocation.dynamic_allocation_policy import dynamic_allocation


OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "experiment_09_dynamic_allocation_policy"
)

OUTPUT_FILE = OUTPUT_DIR / "dynamic_allocation_results.json"


def get_sample_gabriel_rank_output():
    """
    Sample rank output based on Gabriel's Project 1 configuration.

    Gabriel's config:
        batch 16  -> max rank 4
        batch 64  -> max rank 8
        batch 256 -> max rank 16

    In the full integration, this dictionary should be replaced by
    Gabriel's actual rank-selection output file.
    """
    return {
        "client_0": 4,
        "client_1": 8,
        "client_2": 16,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rank_assignments = get_sample_gabriel_rank_output()

    allocations = dynamic_allocation(
        rank_assignments=rank_assignments,
        alpha=64,
    )

    report = {
        "experiment": "Experiment 09: Dynamic Allocation Policy",
        "description": (
            "First version of the full dynamic allocation policy "
            "connecting Gabriel's adaptive rank output with alpha/scaling."
        ),
        "rank_source": "Gabriel Project 1 adaptive rank-selection output",
        "alpha_policy": "global_alpha_64",
        "allocation_equation": {
            "rank": "r_i = GabrielRankPolicy(s(G_i), batch_i)",
            "alpha": "alpha_i = 64",
            "scaling": "scaling_i = alpha_i / r_i",
        },
        "rank_assignments": rank_assignments,
        "dynamic_allocations": allocations,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print("EXPERIMENT 09 COMPLETE")
    print("=" * 60)
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nDynamic Allocations:")
    for client_id, allocation in allocations.items():
        print(client_id, allocation)


if __name__ == "__main__":
    main()
