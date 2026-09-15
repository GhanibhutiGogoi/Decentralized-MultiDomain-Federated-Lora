"""Scientific invariants of domain-weighted gossip and the P2 feature bridge."""

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from src.federated.domain_weights import (
    DOMAIN_POLICY_PROVENANCE, conservative_domain_factors, domain_features,
    features_from_updates, weighted_metropolis,
)
from src.federated.mixing import build_topology, metropolis_hastings


ROOT = Path(__file__).resolve().parents[2]


def canonical_module(relative_path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_peer_weighting_has_independently_derived_transition_probabilities():
    # Uniform proposal and stationary mass 1/4, 3/4. Flow from peer 0 to peer 1
    # is 1/8, so the reverse transition must be (1/8)/(3/4) = 1/6.
    proposal = np.full((2, 2), 0.5)
    expected = np.array([[0.5, 0.5], [1.0 / 6.0, 5.0 / 6.0]])
    p = weighted_metropolis(proposal, [1, 3])
    np.testing.assert_allclose(p, expected, atol=1e-15)
    assert not np.allclose(p, proposal)
    assert not np.allclose(p.sum(axis=0), 1.0)
    values = np.array([0.0, 8.0])
    for _ in range(100):
        values = p @ values
    # Weighted consensus is 6; a Sinkhorn-normalized no-op would converge to 4.
    np.testing.assert_allclose(values, [6.0, 6.0], atol=1e-12)


@pytest.mark.parametrize("topology", ["ring", "star", "path", "fully_connected"])
def test_weighted_stationarity_detailed_balance_and_graph_support(topology):
    proposal = metropolis_hastings(build_topology(list(range(8)), topology))
    weights = np.array([1, 3, 2, 9, 7, 4, 5, 12], dtype=float)
    pi = weights / weights.sum()
    p = weighted_metropolis(proposal, weights)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-14)
    np.testing.assert_allclose(pi @ p, pi, atol=1e-14)
    flow = pi[:, None] * p
    np.testing.assert_allclose(flow, flow.T, atol=1e-14)
    off_diagonal = ~np.eye(len(p), dtype=bool)
    np.testing.assert_array_equal(p[off_diagonal] > 0, proposal[off_diagonal] > 0)
    assert np.all(p >= 0)
    assert np.max(np.abs(p - proposal)) > 0.01
    np.testing.assert_allclose(weighted_metropolis(proposal, np.ones(8)), proposal, atol=1e-15)
    np.testing.assert_allclose(weighted_metropolis(proposal, weights * 1e300), p, atol=1e-15)
    initial = np.arange(8.0) ** 2
    expected = pi @ initial
    values = initial.copy()
    # Weighted star leaves depart slowly when the center has low target mass;
    # 5000 rounds allows this valid kernel to approach its weighted consensus.
    for _ in range(5000):
        values = p @ values
    np.testing.assert_allclose(pi @ values, expected, atol=1e-10)
    np.testing.assert_allclose(values, expected, atol=1e-7)


@pytest.mark.parametrize("weights", [[0, 1], [-1, 2], [1], [1, np.inf], [1, np.nan], [[1, 2]]])
def test_invalid_target_weights_are_rejected(weights):
    with pytest.raises(ValueError, match="weights"):
        weighted_metropolis(np.full((2, 2), 0.5), weights)


@pytest.mark.parametrize("proposal", [[], [1, 2], [[1, 0, 0], [0, 1, 0]],
                                      [[0.5, 0.5], [0.2, 0.8]],
                                      [[0.5, 0.2], [0.2, 0.5]],
                                      [[1.1, -0.1], [-0.1, 1.1]],
                                      [[np.nan, 0], [0, 1]]])
def test_invalid_proposals_are_rejected(proposal):
    with pytest.raises(ValueError, match="proposal"):
        weighted_metropolis(proposal, [1, 2])


def test_policy_is_canonical_p2_code_and_provenance_hashes_match():
    canonical = canonical_module(DOMAIN_POLICY_PROVENANCE["source"], "test_p2_policy")
    samples, quality = [2, 5, 8], [0.4, 0.6, 0.5]
    features = {"js_to_global": [0.0, 0.3, 0.01], "normalized_entropy": [0.1, 0.3, 0.9]}
    actual = conservative_domain_factors(samples, quality, features)
    np.testing.assert_array_equal(actual, canonical.conservative_domain_factors(samples, quality, features))
    for path_key, hash_key in [("source", "source_sha256"), ("feature_source", "feature_source_sha256")]:
        assert hashlib.sha256((ROOT / DOMAIN_POLICY_PROVENANCE[path_key]).read_bytes()).hexdigest() == DOMAIN_POLICY_PROVENANCE[hash_key]
    assert conservative_domain_factors.__code__.co_filename == str(ROOT / DOMAIN_POLICY_PROVENANCE["source"])
    np.testing.assert_allclose(np.average(actual, weights=np.multiply(samples, quality)), 1.0, atol=1e-14)
    assert np.ptp(actual) > 0.01
    assert np.min(actual) >= 0.85 and np.max(actual) <= 1.15
    proposal = np.full((3, 3), 1.0 / 3)
    plain = weighted_metropolis(proposal, np.multiply(samples, quality))
    domain = weighted_metropolis(proposal, np.multiply(samples, quality) * actual)
    # Prevent regression to the earlier diagonal-scale + Sinkhorn no-op.
    assert np.max(np.abs(plain - domain)) > 0.001


def test_feature_definitions_match_p2_measurements_including_missing_classes():
    signals = canonical_module(DOMAIN_POLICY_PROVENANCE["feature_source"], "test_p2_signals")
    histograms = np.array([[4, 0, 1, 0], [1, 3, 2, 1], [2, 2, 2, 2]])
    updates = [np.array([1, 0, -2.0]), np.array([0, 1.0]), np.zeros(3)]
    actual = features_from_updates(updates, histograms)
    update_records = signals.update_dissimilarity_records("fixture", 1, updates)
    labels, indices = [], []
    for counts in histograms:
        start = len(labels)
        labels.extend(np.repeat(np.arange(4), counts).tolist())
        indices.append(list(range(start, len(labels))))
    label_records, _ = signals.label_distribution_records("fixture", labels, indices, 4)
    for name in actual:
        records = update_records if name.startswith("update_") else label_records
        np.testing.assert_allclose(actual[name], [record[name] for record in records], atol=1e-14)
    assert actual["class_imbalance_ratio"][0] == 6.0
    assert actual["normalized_entropy"][2] == 1.0
    assert actual["update_cosine_distance_to_mean"][2] == 0.0


def test_effective_update_bridge_subtracts_reference_and_handles_heterogeneous_ranks():
    def state(rank, magnitude):
        a = torch.zeros(rank, 2, dtype=torch.float64)
        b = torch.zeros(2, rank, dtype=torch.float64)
        a[0, 0], b[0, 0] = 1.0, magnitude * rank / 16.0
        return {"fc": {"A": a, "B": b}}
    before = {12: state(1, 100), 8: state(3, 200), 30: state(2, 300)}
    after = {12: state(2, 101), 8: state(1, 204), 30: state(4, 298)}
    histograms = {30: [2, 3], 12: [1, 1], 8: [2, 1]}
    actual = domain_features(after, histograms, 16, reference_states=before)
    expected = features_from_updates([[1, 0, 0, 0], [4, 0, 0, 0], [-2, 0, 0, 0]],
                                     [histograms[12], histograms[8], histograms[30]])
    for name in actual:
        np.testing.assert_allclose(actual[name], expected[name], atol=1e-14)
    cumulative = domain_features(after, histograms, 16)
    assert not np.allclose(actual["update_l2_distance_to_mean"], cumulative["update_l2_distance_to_mean"])


@pytest.mark.parametrize("histograms", [[[1, 0]], [[1, 0], [0, 0]], [[1, 0], [1, -1]],
                                        [[1, 0], [np.nan, 1]], [[1], [np.inf]]])
def test_invalid_histograms_fail_before_producing_allocation(histograms):
    with pytest.raises(ValueError, match="histograms"):
        features_from_updates([[1, 2], [3, 4]], histograms)


def test_invalid_update_vectors_are_rejected():
    for updates in [[], [np.array([])], [np.array([np.inf])], [np.array([np.nan])]]:
        with pytest.raises(ValueError, match="updates"):
            features_from_updates(updates, [[1, 2]])
