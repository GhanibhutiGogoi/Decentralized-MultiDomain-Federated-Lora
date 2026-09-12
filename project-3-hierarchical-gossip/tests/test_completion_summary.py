"""Guard honest aggregation: sample SD, paired signs, incompatible contexts."""

import json
import math

import pytest

from experiments import summarize_completion as summary


def entry(variant, seed, accuracy, split="same", context="context"):
    record = {"summary": {"personalized_accuracy": accuracy, "consensus_accuracy": accuracy / 2},
              "split_sha256": split, "initial_state_sha256": "same", "rounds": []}
    return {"context_id": context, "variant_id": variant, "label": variant,
            "variant": {"method": variant}, "seed": seed, "record": record,
            "directory": "source", "classification": "full_data_benchmark"}


def test_statistics_use_sample_sd_and_preserve_negative_differences():
    entries = [entry("a", 1, 0.6), entry("a", 2, 0.8), entry("b", 1, 0.5), entry("b", 2, 0.6)]
    groups, aggregates, *_ = summary.aggregate(entries)
    first = aggregates[0]["metrics"]["personalized_accuracy"]
    assert first["mean"] == pytest.approx(0.7)
    assert first["sample_std"] == pytest.approx(math.sqrt(0.02))
    rows, comparisons, exclusions = summary.paired_differences(groups)
    assert not exclusions
    differences = [row["difference_right_minus_left"] for row in rows if row["metric"] == "personalized_accuracy"]
    assert differences == pytest.approx([-0.1, -0.2])
    aggregate = next(row for row in comparisons if row["metric"] == "personalized_accuracy")
    assert aggregate["mean"] == pytest.approx(-0.15)
    assert aggregate["n"] == 2
    assert summary.statistics_for([1])["sample_std"] is None


def test_pairing_requires_identical_split_and_initialization():
    entries = [entry("a", 1, 0.6), entry("a", 2, 0.8), entry("b", 1, 0.5, split="other")]
    groups, *_ = summary.aggregate(entries)
    rows, _, exclusions = summary.paired_differences(groups)
    assert not rows
    assert {item["reason"] for item in exclusions} == {
        "seed missing from one arm", "initialization or split checksum mismatch"}


def test_context_separates_training_but_ignores_reporting_source_additions():
    manifest = {"config": {"ranks": [8], "rounds": 30, "lr": 0.001},
                "source": {"files_sha256": {"experiments/protocol_benchmark.py": "a"}}}
    original, _ = summary.context_for(manifest)
    manifest["source"]["files_sha256"]["experiments/summarize_completion.py"] = "new-report"
    assert summary.context_for(manifest)[0] == original
    manifest["config"]["lr"] = 0.002
    assert summary.context_for(manifest)[0] != original


def test_duplicate_arm_seed_is_rejected_instead_of_counted_twice(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    manifest = {"config": {"ranks": [8], "rounds": 1, "seeds": [42], "methods": ["local"]}}
    record = {"status": "complete", "rounds": [{"round": 1}], "summary": {},
              "seed": 42, "method": "local", "merge": "delta", "error_feedback": False}
    (source / "manifest.json").write_text(json.dumps(manifest))
    (source / "results.jsonl").write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="double-count"):
        summary.load_inputs([source, source])
