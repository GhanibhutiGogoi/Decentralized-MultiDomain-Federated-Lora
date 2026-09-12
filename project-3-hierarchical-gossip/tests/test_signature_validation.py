"""Check that the signature experiment cannot use scoring labels to cluster."""

import importlib

import torch


experiment = importlib.import_module("experiments.05_signature_validation")


def test_scoring_labels_do_not_change_predicted_clusters():
    states = {}
    a = torch.tensor([[1.0, -2.0, 0.5], [0.5, 1.0, -1.0]])
    for cid in range(6):
        b = torch.zeros(6, 2)
        first_row = 0 if cid < 3 else 4
        b[first_row, 0] = 1
        b[first_row + 1, 1] = 1
        states[cid] = {"fc": {"A": a, "B": b}}
    truth = {cid: cid // 3 for cid in states}
    unrelated = {cid: cid % 2 for cid in states}
    first = experiment.score_signatures(states, truth, alpha=32, n_clusters=2, seed=42, stage=2)
    second = experiment.score_signatures(states, unrelated, alpha=32, n_clusters=2, seed=42, stage=2)
    for original, changed in zip(first, second):
        assert original["assignments"] == changed["assignments"]
        assert original["clustering_client_order"] == changed["clustering_client_order"]
        if original["signature"] != "spectral_baseline":
            assert original["adjusted_rand_index"] == 1.0
            assert changed["adjusted_rand_index"] < original["adjusted_rand_index"]


def test_signature_schedule_controls_training_and_defaults_to_nonoracle_modes():
    config = experiment.parse_args(["--output", "unused", "--stages", "2", "5", "10"])
    assert config.rounds == 10
    assert config.methods == ["local", "mh"]
    assert config.stages == [2, 5, 10]
