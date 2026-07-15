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

    The default real-signal run intentionally reports whether the allocation
    collapses to uniform ranks. A separate heterogeneity probe is included to
    show that the formula can produce heterogeneous ranks when input signals
    are sufficiently dispersed.
"""

import json
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


def amplify_signal_spread(client_features, factor=20.0):
    """
    Create a heterogeneity probe by amplifying the spread of real signals.

    This does not replace the default real-signal run. It is only used to show
    that the weighted assignment formula can produce heterogeneous ranks when
    client/domain signals are sufficiently dispersed.
    """
    amplified = {
        client_id: dict(features)
        for client_id, features in client_features.items()
    }

    signal_keys = [
        "complexity_score",
        "entropy",
        "data_imbalance",
    ]

    for key in signal_keys:
        values = [
            float(features.get(key, 0.0))
            for features in client_features.values()
        ]

        mean_value = sum(values) / len(values)

        for client_id, features in amplified.items():
            original_value = float(features.get(key, 0.0))
            amplified_value = mean_value + factor * (
                original_value - mean_value
            )
            amplified[client_id][key] = max(amplified_value, 0.0)

    return amplified


def summarize_rank_counts(results):
    """
    Count how many clients receive each rank.
    """
    rank_counts = {}

    for allocation in results["allocations"].values():
        rank = str(allocation["rank"])
        rank_counts[rank] = rank_counts.get(rank, 0) + 1

    return rank_counts


def summarize_raw_rank_range(results):
    """
    Report the min and max raw continuous rank before nearest-rank mapping.
    """
    raw_ranks = [
        allocation["raw_rank"]
        for allocation in results["allocations"].values()
    ]

    return {
        "min_raw_rank": min(raw_ranks),
        "max_raw_rank": max(raw_ranks),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    client_features = load_client_features()

    # Matched-budget reference:
    # Uniform rank 16 over 15 clients gives total_rank_budget = 240.
    total_rank_budget = 240

    default_results = weighted_assignment(
        client_features=client_features,
        total_rank_budget=total_rank_budget,
        allowed_ranks=[4, 8, 16, 32, 64],
        min_rank=4,
        alpha=64,
        lambda_complexity=1.0,
        lambda_entropy=0.5,
        lambda_imbalance=0.5,
    )

    # Heterogeneity probe:
    # Use the same real client/domain signals, but amplify their spread.
    # This is not a final training configuration. It is only a stress test
    # showing that the formula can produce non-uniform ranks when signals are
    # sufficiently dispersed.
    signal_spread_factor = 50.0

    amplified_features = amplify_signal_spread(
        client_features=client_features,
        factor=signal_spread_factor,
    )

    heterogeneity_probe_results = weighted_assignment(
        client_features=amplified_features,
        total_rank_budget=total_rank_budget,
        allowed_ranks=[4, 8, 16, 32, 64],
        min_rank=4,
        alpha=64,
        lambda_complexity=1.0,
        lambda_entropy=0.5,
        lambda_imbalance=0.5,
    )

    output = {
        "default_matched_budget_240": {
            "description": (
                "Default run using real Experiment 08 client/domain signals "
                "and total_rank_budget=240."
            ),
            "interpretation": (
                "On this data and budget, the weighted assignment collapses "
                "to uniform rank-16 allocation because the client weights are "
                "all close to 1/15 and raw ranks round to 16 under the coarse "
                "rank ladder [4, 8, 16, 32, 64]."
            ),
            "rank_counts": summarize_rank_counts(default_results),
            "raw_rank_range": summarize_raw_rank_range(default_results),
            "results": default_results,
        },
        "heterogeneity_probe": {
            "description": (
                "Probe run using amplified spread of the same client/domain "
                "signals to verify that the formula can produce heterogeneous "
                "rank assignments when signals are more dispersed."
            ),
            "interpretation": (
                "This probe is a formula stress test, not a final training "
                "configuration. It demonstrates that the weighted assignment "
                "formula can produce non-uniform ranks when the input signals "
                "are sufficiently separated relative to the rank ladder."
            ),
            "signal_spread_factor": signal_spread_factor,
            "rank_counts": summarize_rank_counts(heterogeneity_probe_results),
            "raw_rank_range": summarize_raw_rank_range(
                heterogeneity_probe_results
            ),
            "results": heterogeneity_probe_results,
        },
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print("=" * 80)
    print("Experiment 10: Weighted Assignment Policy")
    print("=" * 80)
    print(f"Output saved to: {OUTPUT_FILE}")
    print()
    print("Default matched-budget run:")
    print("  Rank counts:", output["default_matched_budget_240"]["rank_counts"])
    print("  Raw rank range:", output["default_matched_budget_240"]["raw_rank_range"])
    print()
    print("Heterogeneity probe:")
    print("  Signal spread factor:", signal_spread_factor)
    print("  Rank counts:", output["heterogeneity_probe"]["rank_counts"])
    print("  Raw rank range:", output["heterogeneity_probe"]["raw_rank_range"])
    print("=" * 80)
    print("Experiment 10 complete!")


if __name__ == "__main__":
    main()
