"""Tiny real-training guards for the pooled-versus-peer experiment contract.

Synthetic tensors avoid downloads. These checks exercise real LoRA training,
P1 decisions, P2 weighting, neighbor mixing and final deployment assembly.
Run on the designated SSH test machine with the rest of the P3 test suite.
"""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments import integrated_benchmark as benchmark
from experiments.validate_integrated_results import validate_records
from src.federated.merge import lora_to_delta


def _tiny_fixture():
    rng = torch.Generator().manual_seed(892)
    labels = torch.tensor([0] * 6 + [1, 2, 1, 2, 1, 2, 1, 2, 1]
                          + [3, 4, 5, 6] * 3)
    train = {"features": torch.randn(len(labels), 20, generator=rng), "labels": labels}
    test = {"features": torch.randn(12, 20, generator=rng),
            "labels": torch.tensor([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4])}
    partitions = (list(range(6)), list(range(6, 15)), list(range(15, 27)))
    splits = {
        cid: {"domain_id": cid, "train_indices": indices,
              "test_indices": list(range(4 * cid, 4 * (cid + 1))),
              "train_class_counts": torch.bincount(labels[indices], minlength=100).tolist()}
        for cid, indices in enumerate(partitions)
    }
    config = SimpleNamespace(reference_rank=16, alpha=32.0, device="cpu",
                             topology="path", batch_size=8, eval_batch_size=8,
                             local_epochs=1, lr=0.001, weight_decay=0.0001,
                             rounds=4, max_train_per_client=0,
                             max_test_per_domain=0)
    metadata = {"cache_identity_sha256": "synthetic-integration-fixture"}
    return config, train, test, splits, metadata


@pytest.fixture(scope="module")
def paired_runs(tmp_path_factory):
    config, train, test, splits, metadata = _tiny_fixture()
    folder = tmp_path_factory.mktemp("paired-integrated")
    records, paths = {}, {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(benchmark, "make_splits", lambda *args: (copy.deepcopy(splits), {i: i for i in splits}))
        for method in benchmark.METHODS:
            output = folder / method
            output.mkdir()
            benchmark.run_one(copy.copy(config), 42, method, train, test, metadata, output)
            records[method] = json.loads((output / f"seed42_{method}.json").read_text())
            paths[method] = output / f"seed42_{method}_adapter.pt"
    return config, train, test, splits, records, paths


def test_all_arms_share_scaling_initialization_partition_and_exposure(paired_runs):
    config, train, test, _, records, _ = paired_runs
    assert len({r["initial_state_sha256"] for r in records.values()}) == 1
    assert len({r["split_sha256"] for r in records.values()}) == 1
    assert len({r["feature_cache_identity_sha256"] for r in records.values()}) == 1
    for record in records.values():
        assert record["status"] == "complete"
        assert record["alpha"] == config.alpha
        assert record["reference_rank"] == config.reference_rank
        assert record["n_train"] == len(train["labels"])
        assert record["n_test"] == len(test["labels"])
        assert record["training_sample_exposures"] == config.rounds * len(train["labels"]) * config.local_epochs
        assert all(row["train_sample_exposures"] == len(train["labels"]) * config.local_epochs
                   for row in record["rounds"])
        assert record["optimizer"]["reset_each_round"] == (record["method"] != "pooled")


def test_adaptive_warmup_matches_fixed_policy_without_rng_perturbation(paired_runs):
    _, _, _, _, records, _ = paired_runs
    for suffix in ("quality", "domain"):
        fixed, adaptive = records[f"fixed_{suffix}"], records[f"adaptive_{suffix}"]
        for fixed_row, row in zip(fixed["rounds"][:2], adaptive["rounds"][:2]):
            assert row["ranks"] == fixed_row["ranks"] == adaptive["capacity_ceilings"]
            assert row["train_loss"] == fixed_row["train_loss"]
            np.testing.assert_allclose(row["weights"], fixed_row["weights"], atol=1e-12, rtol=0)
            assert row["full_test_correct"] == fixed_row["full_test_correct"]
        for row in adaptive["rounds"]:
            for cid, rank in row["ranks"].items():
                ceiling = adaptive["capacity_ceilings"][cid]
                assert ceiling // 2 <= rank <= ceiling
                assert row["controller_diagnostics"][cid]["rounds_seen"] == row["round"]
            assert row["gradient_probe_examples"] > 0
            assert row["gradient_probe_rank_sample_products"] > 0
            assert row["quality_probe_examples"] > 0


def test_domain_signal_changes_actual_mixer_and_control_traffic_is_charged(paired_runs):
    _, _, _, _, records, _ = paired_runs
    quality = records["fixed_quality"]["rounds"][0]
    domain = records["fixed_domain"]["rounds"][0]
    # Same pre-mixing training states: a real domain policy must change the
    # next update, guarding against the earlier Sinkhorn cancellation defect.
    np.testing.assert_allclose(quality["quality"], domain["quality"], atol=1e-12, rtol=0)
    assert np.max(np.abs(np.asarray(domain["domain_factors"]) - 1)) > 1e-7
    assert np.max(np.abs(np.asarray(domain["weights"]) - quality["weights"])) > 1e-7
    assert np.max(np.abs(np.asarray(domain["mixing_matrix"]) - quality["mixing_matrix"])) > 1e-7
    assert domain["control_bytes"] > quality["control_bytes"] > 0
    assert domain["control_messages"] > 0
    assert records["pooled"]["total_control_bytes"] == 0


def test_peer_matrices_preserve_logged_weights_and_assembly_follows_graph(paired_runs):
    _, _, _, _, records, _ = paired_runs
    for method, record in records.items():
        if method.startswith("pooled"):
            continue
        graph = record["topology"]
        for row in record["rounds"]:
            matrix, weights = np.asarray(row["mixing_matrix"]), np.asarray(row["weights"])
            np.testing.assert_allclose(matrix.sum(1), 1, atol=1e-12, rtol=0)
            np.testing.assert_allclose(weights @ matrix, weights, atol=1e-12, rtol=0)
            if method == "fedavg16":
                assert row["assembly"] is None
                continue
            if method != "fedavg16":
                for i in range(len(weights)):
                    for j in range(len(weights)):
                        if i != j and j not in graph[str(i)]:
                            assert matrix[i, j] == 0
            assert row["assembly"]["messages"] == len(weights) - 1
            assert row["assembly"]["bytes"] > 0
            for edge in row["assembly"]["tree_edges"]:
                assert edge["receiver"] in graph[str(edge["sender"])]


def test_saved_rank16_adapter_reproduces_full_test_endpoint(paired_runs):
    config, _, test, _, records, paths = paired_runs
    for method, path in paths.items():
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        assert artifact["alpha"] == config.alpha
        assert artifact["state"]["fc"]["A"].shape[0] == config.reference_rank
        initial, state = artifact["initial"], artifact["state"]
        weight = initial["weight"] + lora_to_delta(state, artifact["alpha"])["fc"]
        prediction = torch.nn.functional.linear(test["features"], weight, initial["bias"]).argmax(1)
        correct = int((prediction == test["labels"]).sum())
        assert correct == records[method]["rounds"][-1]["full_test_correct"]
        assert correct / len(test["labels"]) == records[method]["final_full_test_accuracy"]


def test_artifact_validator_accepts_pairing_and_detects_protocol_drift(paired_runs):
    records = paired_runs[4]
    report = validate_records(list(records.values()))
    assert report["status"] == "passed", report["failed_checks"]
    altered = copy.deepcopy(list(records.values()))
    altered[0]["alpha"] *= 2
    altered[0]["rounds"][0]["train_sample_exposures"] -= 1
    broken = validate_records(altered)
    failed = {item["check"] for item in broken["failed_checks"]}
    assert broken["status"] == "failed"
    assert "paired_alpha" in failed
    assert "each_round_covers_training_split" in failed


def test_test_labels_cannot_change_adaptive_ranks_weights_or_trained_adapter(tmp_path, monkeypatch):
    config, train, test, splits, metadata = _tiny_fixture()
    monkeypatch.setattr(benchmark, "make_splits", lambda *args: (copy.deepcopy(splits), {i: i for i in splits}))
    outputs = []
    for label_shift in (0, 37):
        changed_test = {"features": test["features"], "labels": (test["labels"] + label_shift) % 100}
        output = tmp_path / str(label_shift)
        output.mkdir()
        benchmark.run_one(copy.copy(config), 42, "adaptive_domain", train, changed_test, metadata, output)
        record = json.loads((output / "seed42_adaptive_domain.json").read_text())
        artifact = torch.load(output / "seed42_adaptive_domain_adapter.pt", weights_only=False)
        outputs.append((record, artifact))
    for first, second in zip(outputs[0][0]["rounds"], outputs[1][0]["rounds"]):
        assert first["ranks"] == second["ranks"]
        assert first["weights"] == second["weights"]
        assert first["controller_diagnostics"] == second["controller_diagnostics"]
    for factor in ("A", "B"):
        assert torch.equal(outputs[0][1]["state"]["fc"][factor], outputs[1][1]["state"]["fc"][factor])


@pytest.mark.parametrize("missing", ["train_indices", "test_indices"])
def test_full_data_endpoint_rejects_missing_examples(tmp_path, monkeypatch, missing):
    config, train, test, splits, metadata = _tiny_fixture()
    splits[0][missing] = splits[0][missing][1:]
    monkeypatch.setattr(benchmark, "make_splits", lambda *args: (splits, {i: i for i in splits}))
    with pytest.raises(ValueError, match="all (training|test) examples"):
        benchmark.run_one(config, 42, "pooled", train, test, metadata, tmp_path)
