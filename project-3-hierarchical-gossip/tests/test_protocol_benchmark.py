"""Protocol invariants for the frozen-feature experiment (no dataset download)."""

import json

import numpy as np
import pytest
import torch

from experiments import protocol_benchmark as benchmark
from src.federated.hierarchical import is_doubly_stochastic
from src.federated.merge import lora_to_delta


def _state(a, b):
    return {"fc": {"A": torch.tensor(a, dtype=torch.float32),
                   "B": torch.tensor(b, dtype=torch.float32)}}


@pytest.mark.parametrize('method', benchmark.SUPERSEDED_METHODS)
def test_invalid_historical_composition_cannot_be_rerun_as_evidence(method):
    with pytest.raises(ValueError, match='integrated_benchmark'):
        benchmark.build_mixing(method, {0: 0, 1: 1}, 5)


def test_flat_ring_is_seeded_independently_of_domain_order():
    assignments = {i: i // 3 for i in range(15)}
    first = benchmark.build_mixing("mh", assignments, 5, seed=42)(0)
    again = benchmark.build_mixing("mh", assignments, 5, seed=42)(0)
    other = benchmark.build_mixing("mh", assignments, 5, seed=43)(0)
    assert is_doubly_stochastic(first)
    assert np.allclose(first, first.T)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)
    assert benchmark.topology_order(assignments, 42) != sorted(assignments)
    assert np.all(np.count_nonzero(first - np.diag(np.diag(first)), axis=1) == 2)


def test_zero_padding_has_heterogeneous_receiver_shapes_and_gauge_error():
    # Same represented update, opposite gauges: naive factor averaging erases it.
    first = _state([[1, 0]], [[1], [0]])
    # alpha/r changes with rank, so the second product is 2 and represents the
    # same effective delta (alpha=2: rank-1 product 1 -> delta 2; rank-2
    # product 2 -> delta 2) under a different factor gauge.
    second = _state([[-2, 0], [0, 0]], [[-1, 0], [0, 0]])
    mixed = benchmark.zero_pad_factor_mix([first, second], np.full((2, 2), 0.5), [1, 2])
    assert mixed[0]["fc"]["A"].shape == (1, 2)
    assert mixed[1]["fc"]["A"].shape == (2, 2)
    assert torch.count_nonzero(mixed[0]["fc"]["B"]) == 0
    expected = (lora_to_delta(first, 2)["fc"] + lora_to_delta(second, 2)["fc"]) / 2
    assert not torch.allclose(lora_to_delta(mixed[0], 2)["fc"], expected)


def test_shared_initial_base_is_independent_of_rank():
    low = benchmark.initial_parameters(8, 2, 42)
    high = benchmark.initial_parameters(8, 6, 42)
    assert torch.equal(low["weight"], high["weight"])
    assert torch.equal(low["bias"], high["bias"])
    assert torch.equal(low["A"], high["A"][:2])


def test_operational_cost_accounts_for_exact_bridge_intermediates():
    states = [_state([[1, 0, 0]], [[1], [0]]) for _ in range(4)]
    assignments = {0: 0, 1: 0, 2: 1, 3: 1}
    diag = {"messages": 12, "floats": 60}
    quiet = benchmark.operational_cost("oracle", assignments, 0, 2, states, diag)
    bridge = benchmark.operational_cost("oracle", assignments, 1, 2, states, diag)
    fedavg = benchmark.operational_cost("fedavg", assignments, 0, 2, states, diag)
    assert quiet[:2] == (4, 20)
    # Four directed intra edges twice and two representative messages. First
    # stage sends 5-factor floats; later stages send exact 2x3 dense updates.
    assert bridge[:2] == (10, 20 + 6 * 6)
    assert fedavg[:2] == (8, 20 + 4 * 6)


def _fake_features():
    generator = torch.Generator().manual_seed(5)
    train = {"features": torch.randn(400, 8, generator=generator),
             "labels": torch.arange(100).repeat_interleave(4)}
    test = {"features": torch.randn(200, 8, generator=generator),
            "labels": torch.arange(100).repeat_interleave(2)}
    metadata = {key: {"sha256": f"synthetic-{key}"} for key in ("train", "test")}
    return train, test, metadata


def test_small_battery_writes_paired_complete_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "load_features", lambda *args, **kwargs: _fake_features())
    output = tmp_path / "run"
    manifest = benchmark.main(["--output", str(output), "--device", "cpu", "--rounds", "1",
                               "--seeds", "7", "--n-domains", "2", "--clients-per-domain", "2",
                               "--ranks", "2", "4", "--bridge-every", "1", "--torch-threads", "1"])
    assert manifest["status"] == "complete"
    assert manifest["run_classification"] == "smoke"
    records = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert len(records) == 4
    assert len({row["initial_state_sha256"] for row in records}) == 1
    assert len({row["split_sha256"] for row in records}) == 1
    for row in records:
        assert row["status"] == "complete"
        metrics = row["rounds"][0]
        assert 0 <= metrics["personalized_accuracy"] <= 1
        assert 0 <= metrics["consensus_accuracy"] <= 1
        assert metrics["wall_seconds"] > 0
        assert len(metrics["per_domain_accuracy"]) == 2
    assert records[0]["summary"]["total_effective_messages"] == 0
    assert records[1]["summary"]["total_operational_messages"] == 8
    assert len((output / "summary.csv").read_text().splitlines()) == 5
    with pytest.raises(FileExistsError, match="empty"):
        benchmark.main(["--output", str(output)])


def test_factor_battery_labels_error_separately_from_svd_tail(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "load_features", lambda *args, **kwargs: _fake_features())
    output = tmp_path / "factor"
    benchmark.main(["--output", str(output), "--device", "cpu", "--rounds", "1", "--seeds", "7",
                    "--methods", "mh", "--n-domains", "2", "--clients-per-domain", "2",
                    "--ranks", "2", "4", "--merge", "factor_zero_pad", "--torch-threads", "1"])
    record = json.loads((output / "seed7_mh.json").read_text())
    assert record["rounds"][0]["mean_tail_mass"] is None
    assert record["rounds"][0]["mean_residual_energy"] is None
    assert record["rounds"][0]["mean_merge_error_energy"] >= 0


def test_rejects_invalid_feedback_combination_and_duplicate_runs():
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--output", "unused", "--merge", "factor_zero_pad", "--error-feedback"])
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--output", "unused", "--seeds", "42", "42"])
