"""Statistical and provenance guards for the corrected paper summary."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("integrated_summary", ROOT / "scripts/summarize_integrated.py")
summary_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary_tool)


def records_fixture():
    records = []
    for seed_index, seed in enumerate((42, 43, 44)):
        for method_index, method in enumerate(summary_tool.METHODS):
            pooled = method.startswith("pooled")
            adaptive = method.startswith("adaptive")
            accuracy = 0.50 + 0.10 * seed_index - method_index * .01 * (seed_index + 1)
            factor_floats = 0 if pooled else 50 if adaptive else 100
            control = 0 if pooled else 80 if method.endswith("domain") else 8
            assembly = None if pooled or method == "fedavg16" else {"bytes": 120, "peak_peer_dense_bytes": 80,
                                                                   "relative_truncation_energy": 0.1}
            rows = []
            for round_id in (1, 2):
                value = accuracy - .01 * (2 - round_id)
                rows.append({"round": round_id, "full_test_accuracy": value,
                             "full_test_correct": round(value * 10000), "train_sample_exposures": 50000,
                             "train_rank_sample_products": 50000 * (8 if adaptive else 16),
                             "training_factor_floats": factor_floats, "control_bytes": control,
                             "train_loss": 1.0, "train_seconds": 1.0, "assembly": assembly,
                             "ranks": {} if pooled else {"0": 4, "1": 8, "2": 16},
                             "gradient_probe_examples": 24 if adaptive else 0,
                             "quality_probe_examples": 24 if "quality" in method or "domain" in method else 0})
            records.append({"schema_version": 2, "status": "complete", "method": method, "seed": seed,
                            "alpha": 32.0, "reference_rank": 16,
                            "initial_state_sha256": f"initial-{seed}", "split_sha256": f"split-{seed}",
                            "feature_cache_identity_sha256": "same-real-features", "n_train": 50000, "n_test": 10000,
                            "training_sample_exposures": 100000, "capacity_ceilings": {"0": 4, "1": 8, "2": 16},
                            "optimizer": {"name": "Adam", "lr": .001, "weight_decay": .0001,
                                          "batch_size": 128, "local_epochs": 1, "reset_each_round": method != "pooled"},
                            "rounds": rows, "final_full_test_accuracy": accuracy,
                            "total_training_factor_floats": 2 * factor_floats,
                            "total_control_bytes": 2 * control, "final_assembly": assembly,
                            "total_deployment_payload_bytes": 8 * factor_floats + 2 * control + (assembly or {}).get("bytes", 0),
                            "evaluation_only_assembly_bytes": (assembly or {}).get("bytes", 0),
                            "total_optimizer_steps": 782 if pooled else 810, "wall_seconds": 4.0})
    return records


def test_differences_are_paired_before_sd_and_sd_is_sample_based():
    result = summary_tool.summarize_records(records_fixture())
    rows = {row["method"]: row for row in result["aggregate"]}
    # pooled_reset differs by -1/-2/-3 pp across seeds; SD must be 1 pp,
    # regardless of the 10 pp variation in the pooled accuracy itself.
    difference = rows["pooled_reset"]["paired_difference_vs_pooled_pp"]
    assert difference["mean"] == pytest.approx(-2.0)
    assert difference["sample_sd"] == pytest.approx(1.0)
    assert rows["pooled"]["accuracy_percent"]["sample_sd"] == pytest.approx(10.0)
    assert rows["pooled_reset"]["optimizer_steps"]["mean"] != rows["fixed_quality"]["optimizer_steps"]["mean"]


def test_factorial_contrasts_and_resource_components_are_separate():
    result = summary_tool.summarize_records(records_fixture())
    factorial = result["factorial_aggregate"]
    assert factorial["rank_effect_without_domain_pp"]["mean"] == pytest.approx(-4)
    assert factorial["domain_effect_fixed_rank_pp"]["mean"] == pytest.approx(-2)
    assert factorial["interaction_pp"]["mean"] == pytest.approx(0)
    resource = [r for r in result["adaptive_resource_pairs"] if r["policy"] == "domain"]
    assert all(r["training_factor_saving_percent"] == 50 for r in resource)
    assert all(r["deployment_payload_saving_percent"] < 50 for r in resource)
    assert all(r["rank_sample_product_saving_percent"] == 50 for r in resource)
    adaptive = [row for row in result["per_seed"] if row["method"] == "adaptive_domain"][0]
    assert adaptive["deployment_payload_bytes"] == 400 + 160 + 120
    assert adaptive["evaluation_only_assembly_bytes"] == 120
    assert adaptive["gradient_probe_examples"] == 48


@pytest.mark.parametrize("field,value", [("status", "running"), ("status", "failed"), ("alpha", 16),
                                         ("initial_state_sha256", "wrong-initial"), ("split_sha256", "wrong-split"),
                                         ("feature_cache_identity_sha256", "wrong-cache"),
                                         ("n_test", 100), ("n_train", 100),
                                         ("training_sample_exposures", 123),
                                         ("total_deployment_payload_bytes", 123),
                                         ("final_full_test_accuracy", .99)])
def test_incomplete_or_mismatched_evidence_is_rejected(field, value):
    records = records_fixture()
    records[1][field] = value
    with pytest.raises(ValueError):
        summary_tool.summarize_records(records)


def test_missing_or_duplicate_run_cannot_silently_change_the_seed_sample():
    records = records_fixture()
    for broken in [records[:-1], records + [copy.deepcopy(records[0])]]:
        with pytest.raises(ValueError, match="grid mismatch|duplicate"):
            summary_tool.summarize_records(broken)


def test_round_count_and_endpoint_count_mismatch_are_rejected():
    records = records_fixture()
    records[1]["rounds"][-1]["full_test_correct"] += 1
    with pytest.raises(ValueError, match="count/accuracy"):
        summary_tool.summarize_records(records)
    records = records_fixture()
    records[1]["rounds"].pop()
    with pytest.raises(ValueError, match="round sequence"):
        summary_tool.summarize_records(records)


def test_single_seed_has_no_invented_error_bar():
    value = summary_tool.stats([42])
    assert value["mean"] == 42
    assert value["sample_sd"] is None


def test_paper_outputs_contain_measured_endpoints_and_qualified_paired_contrasts(tmp_path):
    records = records_fixture()
    result = summary_tool.summarize_records(records)
    latex = summary_tool.latex_table(result)
    assert "No noninferiority or equivalence margin was prespecified" in latex
    assert "Adaptive minus fixed; domain weighted" in latex
    assert "Domain minus quality; adaptive ranks" in latex
    assert r"$-2.00 \pm 1.00$" in latex
    files = summary_tool.figures(result, records, tmp_path)
    assert len(files) == 10
    for name in files:
        assert (tmp_path / name).stat().st_size > 4000
    curves, ranks = summary_tool.curve_tables(records)
    assert len(curves) == 54
    assert len(ranks) == 3 * 7 * 2 * 3
    assert all(np.isfinite(row["full_test_accuracy_percent"]) for row in curves)
