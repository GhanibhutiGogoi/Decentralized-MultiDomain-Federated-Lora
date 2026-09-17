"""Exact data ownership and stratification invariants for quantity-only FL."""

import json

import numpy as np
import pytest

from src.data.quantity_skew import (
    allocate_client_counts, partition_manifest, partition_quantity_skew,
)


def assert_exact_proportional_partition(labels, counts, partitions):
    """Independent integer oracle; no partition implementation helpers used."""
    observed = [index for indices in partitions.values() for index in indices]
    assert sorted(observed) == list(range(len(labels)))
    assert len(observed) == len(set(observed))
    assert [len(partitions[cid]) for cid in range(len(counts))] == list(counts)
    classes, class_counts = np.unique(labels, return_counts=True)
    table = np.array([[sum(labels[index] == cls for index in partitions[cid])
                       for cls in classes] for cid in range(len(counts))])
    np.testing.assert_array_equal(table.sum(0), class_counts)
    np.testing.assert_array_equal(table.sum(1), counts)
    for row, count in enumerate(counts):
        for column, class_count in enumerate(class_counts):
            floor, remainder = divmod(int(count) * int(class_count), len(labels))
            assert table[row, column] in {floor, floor + bool(remainder)}


def test_unequal_quantities_preserve_class_marginals_even_with_a_rare_class():
    labels = np.repeat([4, 7, 21, 99], [13, 7, 4, 1])
    original = labels.copy()
    counts = [2, 5, 7, 11]
    parts = partition_quantity_skew(labels, client_counts=counts, seed=37)
    assert_exact_proportional_partition(labels, counts, parts)
    np.testing.assert_array_equal(labels, original)
    # A class represented once cannot be present in every client; no samples
    # may be duplicated in an attempt to manufacture per-client coverage.
    assert sum(any(labels[index] == 99 for index in shard) for shard in parts.values()) == 1


@pytest.mark.parametrize("seed", [0, 1, 2, 17, 42])
def test_fractional_remainders_preserve_both_marginals_on_varied_tables(seed):
    rng = np.random.default_rng(seed + 1087)
    for _ in range(12):
        class_counts = rng.integers(1, 15, size=6)
        labels = np.repeat(np.arange(6), class_counts)
        cuts = np.sort(rng.choice(np.arange(1, len(labels)), size=6, replace=False))
        counts = np.diff(np.concatenate(([0], cuts, [len(labels)])))
        parts = partition_quantity_skew(labels, client_counts=counts, seed=seed)
        assert_exact_proportional_partition(labels, counts, parts)


def test_largest_remainder_counts_and_ratio_scaling():
    # Quotas 10/6, 20/6, 30/6: the one extra unit belongs to the first client.
    np.testing.assert_array_equal(allocate_client_counts(10, [1, 2, 3]), [2, 3, 5])
    np.testing.assert_array_equal(allocate_client_counts(10, [1e300, 2e300, 3e300]), [2, 3, 5])
    labels = np.tile(np.arange(3), 10)
    counts = allocate_client_counts(30, [.1, .3, .6], seed=9)
    parts = partition_quantity_skew(labels, client_fractions=[.1, .3, .6], seed=9)
    assert_exact_proportional_partition(labels, counts, parts)


def test_rounding_ties_are_repeatable_and_do_not_use_global_rng():
    np.random.seed(871)
    state = np.random.get_state()
    first = allocate_client_counts(10, [1, 1, 1], seed=14)
    again = allocate_client_counts(10, [1, 1, 1], seed=14)
    np.testing.assert_array_equal(first, again)
    assert sorted(first.tolist()) == [3, 3, 4]
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


def test_partition_seed_changes_ownership_without_changing_class_quota_feasibility():
    labels = np.tile(np.arange(4), 20)
    first = partition_quantity_skew(labels, client_counts=[10, 30, 40], seed=42)
    again = partition_quantity_skew(labels, client_counts=[10, 30, 40], seed=42)
    changed = partition_quantity_skew(labels, client_counts=[10, 30, 40], seed=43)
    assert first == again
    assert any(set(first[cid]) != set(changed[cid]) for cid in first)
    assert_exact_proportional_partition(labels, [10, 30, 40], changed)


def test_quantity_assignment_is_independent_of_example_shuffle_seed():
    labels = np.tile(np.arange(4), 9)
    counts = [6, 12, 18]
    first = partition_quantity_skew(labels, client_counts=counts, seed=11, assignment_seed=901)
    other_examples = partition_quantity_skew(labels, client_counts=counts, seed=92, assignment_seed=901)
    assigned = [len(first[cid]) for cid in first]
    assert sorted(assigned) == counts
    assert assigned == [len(other_examples[cid]) for cid in other_examples]
    assert_exact_proportional_partition(labels, assigned, first)
    assert_exact_proportional_partition(labels, assigned, other_examples)


@pytest.mark.parametrize("labels", [["cat", "dog", "cat", "dog"], [-5, 300, -5, 300], [True, False, True, False]])
def test_class_identifiers_need_not_be_contiguous(labels):
    labels = np.asarray(labels)
    parts = partition_quantity_skew(labels, client_counts=[1, 3], seed=7)
    assert_exact_proportional_partition(labels, [1, 3], parts)
    manifest = partition_manifest(labels, parts)
    assert manifest["class_labels"] == np.unique(labels).tolist()
    assert json.loads(json.dumps(manifest, allow_nan=False)) == manifest


def test_one_client_and_one_class_are_valid_edge_cases():
    labels = np.repeat(7, 8)
    one = partition_quantity_skew(labels, client_counts=[8], seed=8)
    several = partition_quantity_skew(labels, client_counts=[1, 2, 5], seed=8)
    assert_exact_proportional_partition(labels, [8], one)
    assert_exact_proportional_partition(labels, [1, 2, 5], several)


def test_manifest_verifies_membership_and_records_recomputable_statistics():
    labels = np.repeat([2, 8, 13], [9, 6, 3])
    parts = partition_quantity_skew(labels, client_counts=[3, 6, 9], seed=17)
    manifest = partition_manifest(labels, parts, seed=17, dataset_id="fixture/train")
    assert manifest["client_counts"] == [3, 6, 9]
    assert manifest["global_class_counts"] == [9, 6, 3]
    assert manifest["coverage"] == {"unique_examples": 18, "missing_examples": 0, "duplicate_examples": 0}
    assert manifest["max_class_quota_deviation"] < 1
    assert manifest["proportional_class_allocation"] is True
    assert sum(client["sample_weight"] for client in manifest["clients"].values()) == pytest.approx(1)
    assert manifest["clients"]["0"]["indices"] == parts[0]
    assert json.loads(json.dumps(manifest, allow_nan=False)) == manifest
    reordered = {cid: list(reversed(indices)) for cid, indices in parts.items()}
    reordered_manifest = partition_manifest(labels, reordered)
    assert manifest["partition_sha256"] == reordered_manifest["partition_sha256"]
    assert manifest["index_order_sha256"] != reordered_manifest["index_order_sha256"]
    swapped = {cid: indices.copy() for cid, indices in parts.items()}
    swapped[0][0], swapped[1][0] = swapped[1][0], swapped[0][0]
    assert partition_manifest(labels, swapped)["partition_sha256"] != manifest["partition_sha256"]
    changed_labels = labels.copy()
    changed_labels[0] = 123
    assert partition_manifest(changed_labels, parts)["label_sha256"] != manifest["label_sha256"]


def test_manifest_does_not_mislabel_a_disjoint_label_partition_as_quantity_only():
    manifest = partition_manifest([0, 0, 1, 1], {0: [0, 1], 1: [2, 3]})
    assert manifest["proportional_class_allocation"] is False
    assert manifest["partition_kind"] == "classification_partition"


@pytest.mark.parametrize("kwargs", [
    {}, {"client_counts": [2, 2], "client_fractions": [1, 1]},
    {"client_counts": [1, 2]}, {"client_counts": [0, 4]},
    {"client_counts": [-1, 5]}, {"client_counts": [2.0, 2.0]},
    {"client_counts": [True, True]}, {"client_counts": [[2, 2]]},
    {"client_fractions": []}, {"client_fractions": [1, 0]},
    {"client_fractions": [1, np.inf]}, {"client_fractions": [1, np.nan]},
    {"client_fractions": [1, -1]}, {"client_fractions": [1, 1, 1, 1, 1]},
    {"client_fractions": [.9999, .0001]},
    {"client_counts": [2, 2], "seed": -1},
    {"client_counts": [2, 2], "seed": True},
    {"client_counts": [2, 2], "assignment_seed": 1.2},
])
def test_invalid_quantity_requests_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        partition_quantity_skew([0, 0, 1, 1], **kwargs)


@pytest.mark.parametrize("labels", [[], [[0, 1]], [0.0, 1.0], [0, np.nan], [{"x": 1}]])
def test_invalid_classification_labels_are_rejected(labels):
    with pytest.raises(ValueError, match="labels"):
        partition_quantity_skew(labels, client_fractions=[1])


@pytest.mark.parametrize("parts", [
    {}, {0: [0, 1], 1: [1, 2, 3]}, {0: [0, 0], 1: [1, 2, 3]},
    {0: [0, 1], 1: [2]}, {0: [0, 1], 1: [2, 4]},
    {0: [0, 1], 1: [-1, 2, 3]}, {0: [], 1: [0, 1, 2, 3]},
    {0: [0.0, 1.0], 1: [2, 3]}, {0: [0, 1], 2: [2, 3]},
    {"0": [0, 1], "1": [2, 3]},
])
def test_manifest_rejects_missing_duplicate_out_of_range_or_invalid_indices(parts):
    with pytest.raises(ValueError):
        partition_manifest([0, 0, 1, 1], parts)
