"""Validate recorded pooled/peer pairing and policy execution, without training.

Example: python -m experiments.validate_integrated_results RUN_DIRECTORY
Writes RUN_DIRECTORY/validation.json and returns a failing exit status if any
protocol invariant fails. Accuracy itself is never an acceptance criterion.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def validate_records(records):
    checks, observations = [], []

    def check(name, passed, **context):
        checks.append({"check": name, "passed": bool(passed), **context})

    check("nonempty_run_set", bool(records))
    by_seed, seen = {}, set()
    for record in records:
        key = (record["seed"], record["method"])
        check("unique_seed_method", key not in seen, seed=key[0], method=key[1])
        seen.add(key)
        by_seed.setdefault(record["seed"], {})[record["method"]] = record
        context = {"seed": record["seed"], "method": record["method"]}
        check("completed_run", record["status"] == "complete" and bool(record["rounds"]), **context)
        rounds = record["rounds"]
        if not rounds:
            continue
        epochs = record["optimizer"]["local_epochs"]
        exposures = record["n_train"] * epochs
        check("each_round_covers_training_split", all(row["train_sample_exposures"] == exposures for row in rounds), **context)
        check("reported_total_exposure", record["training_sample_exposures"] == exposures * len(rounds), **context)
        check("consistent_test_accuracy_denominator", all(
            abs(row["full_test_accuracy"] - row["full_test_correct"] / record["n_test"]) < 1e-12
            for row in rounds), **context)
        check("final_endpoint_matches_last_round", record["final_full_test_accuracy"] == rounds[-1]["full_test_accuracy"], **context)
        if record["method"].startswith("pooled"):
            check("pooled_has_no_communication", record["total_training_factor_floats"] == record["total_control_bytes"] == 0, **context)
            continue
        graph = record["topology"]
        for row in rounds:
            row_context = {**context, "round": row["round"]}
            pi, matrix = np.asarray(row["weights"]), np.asarray(row["mixing_matrix"])
            check("stochastic_weighted_mixer", np.allclose(matrix.sum(1), 1, rtol=0, atol=1e-10)
                  and np.allclose(pi @ matrix, pi, rtol=0, atol=1e-10), **row_context)
            ceilings = record["capacity_ceilings"]
            check("client_ranks_within_capability", all(
                0 < rank <= ceilings[cid] for cid, rank in row["ranks"].items()), **row_context)
            if record["method"].startswith("adaptive_"):
                check("adaptive_half_capability_floor", all(
                    rank >= ceilings[cid] // 2 for cid, rank in row["ranks"].items()), **row_context)
                check("adaptive_gradient_probe_accounted", row.get("gradient_probe_examples", 0) > 0
                      and row.get("gradient_probe_rank_sample_products", 0) > 0, **row_context)
                check("controller_observed_current_round", all(
                    item["rounds_seen"] == row["round"] for item in row["controller_diagnostics"].values()), **row_context)
            if record["method"].endswith(("_quality", "_domain")):
                check("weight_control_traffic_accounted", row["control_bytes"] > 0 and row["control_messages"] > 0, **row_context)
            if record["method"] != "fedavg16":
                check("training_uses_neighbor_edges", all(
                    matrix[i, j] == 0 for i in range(len(pi)) for j in range(len(pi))
                    if i != j and j not in graph[str(i)]), **row_context)
                assembly = row["assembly"]
                check("final_assembly_uses_neighbor_tree", assembly["messages"] == len(pi) - 1
                      and all(edge["receiver"] in graph[str(edge["sender"])]
                              for edge in assembly["tree_edges"]), **row_context)

    for seed, arms in by_seed.items():
        for field in ("alpha", "reference_rank", "initial_state_sha256", "split_sha256",
                      "feature_cache_identity_sha256", "n_train", "n_test", "training_sample_exposures"):
            check("paired_" + field, len({arm[field] for arm in arms.values()}) == 1, seed=seed)
        for field in ("lr", "weight_decay", "batch_size", "local_epochs"):
            check("paired_optimizer_" + field, len({arm["optimizer"][field] for arm in arms.values()}) == 1, seed=seed)
        for suffix in ("quality", "domain"):
            fixed, adaptive = arms.get("fixed_" + suffix), arms.get("adaptive_" + suffix)
            if fixed and adaptive:
                for left, right in zip(fixed["rounds"][:2], adaptive["rounds"][:2]):
                    check("warmup_identical_fixed_and_adaptive", left["ranks"] == right["ranks"]
                          and left["full_test_correct"] == right["full_test_correct"]
                          and np.allclose(left["weights"], right["weights"], rtol=0, atol=1e-12)
                          and np.allclose(left["mixing_matrix"], right["mixing_matrix"], rtol=0, atol=1e-12),
                          seed=seed, suffix=suffix, round=left["round"])
        quality, domain = arms.get("fixed_quality"), arms.get("fixed_domain")
        if quality and domain and quality["rounds"] and domain["rounds"]:
            left, right = quality["rounds"][0], domain["rounds"][0]
            factor_change = float(np.max(np.abs(np.asarray(right["domain_factors"]) - 1)))
            weight_change = float(np.max(np.abs(np.asarray(left["weights"]) - right["weights"])))
            mixer_change = float(np.max(np.abs(np.asarray(left["mixing_matrix"]) - right["mixing_matrix"])))
            observations.append({"seed": seed, "first_round_domain_factor_max_deviation": factor_change,
                                 "first_round_domain_weight_max_change": weight_change,
                                 "first_round_domain_mixer_max_change": mixer_change})
            check("nonconstant_domain_factors_are_not_cancelled", factor_change < 1e-10
                  or (weight_change > 1e-12 and mixer_change > 1e-12), seed=seed)
    return {"status": "passed" if all(item["passed"] for item in checks) else "failed",
            "n_runs": len(records), "n_checks": len(checks),
            "failed_checks": [item for item in checks if not item["passed"]],
            "observations": observations, "checks": checks,
            "scope": "Recorded protocol invariants; scientific accuracy parity is evaluated separately."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--write-json", type=Path)
    args = parser.parse_args()
    records = [json.loads(path.read_text()) for path in sorted(args.directory.glob("seed*_*.json"))]
    report = validate_records(records)
    output = args.write_json or args.directory / "validation.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "checks"}, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
