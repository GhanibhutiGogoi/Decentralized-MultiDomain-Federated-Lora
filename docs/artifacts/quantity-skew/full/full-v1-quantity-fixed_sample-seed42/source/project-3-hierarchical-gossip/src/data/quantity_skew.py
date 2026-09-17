"""Quantity-only classification shards with exact integer marginals.

Clients receive different numbers of examples, while every client/class cell
is either the floor or ceiling of its proportional allocation. The partition
is disjoint and exhaustive; rare classes need not occur at every small client.
Only NumPy and the standard library are required.

``seed`` controls rounding ties and within-class example shuffles. Optional
``assignment_seed`` independently permutes quantity slots onto client IDs;
assign hardware rank caps and topology with *different* experiment seeds.
This prevents a sorted size vector from silently coupling size to capability.
No rank, model, evaluation data, or global random state enters this module.
"""

from collections import deque
from collections.abc import Mapping
import hashlib
import json
import numbers

import numpy as np


def _seed(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _labels(values):
    labels = np.asarray(values)
    if labels.ndim != 1 or not labels.size:
        raise ValueError("labels must be a nonempty one-dimensional classification array")
    if labels.dtype.kind not in "biuU":
        raise ValueError("labels must contain integer, boolean, or string class labels")
    return labels


def _counts(values, total):
    array = np.asarray(values)
    if array.ndim != 1 or not array.size or array.dtype.kind not in "iu":
        raise ValueError("client_counts must be a nonempty vector of positive integers")
    counts = [int(value) for value in array]
    if any(value <= 0 for value in counts) or sum(counts) != total:
        raise ValueError("client_counts must be positive and sum exactly to the number of labels")
    return np.asarray(counts, dtype=np.int64)


def allocate_client_counts(total, client_fractions, *, seed=0):
    """Hamilton/largest-remainder allocation of positive relative shares.

    Fractions may be normalized probabilities or positive ratios such as
    ``[1, 2, 4]``. They are normalized internally. Ties use a seeded ordering.
    A share rounding to zero is rejected rather than silently altering the
    requested skew to create a nonempty client.
    """
    if isinstance(total, (bool, np.bool_)) or not isinstance(total, numbers.Integral) or total < 1:
        raise ValueError("total must be a positive integer")
    seed = _seed(seed, "seed")
    try:
        fractions = np.asarray(client_fractions, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("client_fractions must be finite positive relative shares") from error
    if (fractions.ndim != 1 or not fractions.size or not np.isfinite(fractions).all()
            or np.any(fractions <= 0)):
        raise ValueError("client_fractions must be finite positive relative shares")
    if len(fractions) > total:
        raise ValueError("client_fractions would create empty clients")
    # Scaling before summing also supports large but finite ratio inputs.
    normalized = fractions / fractions.max()
    normalized /= normalized.sum()
    quotas = int(total) * normalized
    counts = np.floor(quotas).astype(np.int64)
    missing = int(total) - int(counts.sum())
    if not 0 <= missing <= len(counts):
        raise ValueError("client_fractions could not be apportioned at this numerical scale")
    priority = np.random.default_rng(seed).permutation(len(counts))
    order = priority[np.argsort(-(quotas - counts)[priority], kind="stable")]
    counts[order[:missing]] += 1
    if np.any(counts == 0):
        raise ValueError("client_fractions would create empty clients; increase data or reduce skew")
    return counts


def _stratified_counts(counts, class_counts, rng):
    """Round a proportional table with exact row and column totals.

    Integer products avoid floating-point quota errors. A bipartite augmenting
    path completes the floor table using at most one extra unit per noninteger
    cell. Greedy independent per-class rounding cannot guarantee both totals.
    The fractional remainder table is a feasible flow; integrality of the
    bipartite flow guarantees an integer completion.
    """
    total = sum(map(int, counts))
    products = counts.astype(object)[:, None] * class_counts.astype(object)[None, :]
    allocation = np.asarray(products // total, dtype=np.int64)
    remainder = np.asarray(products % total, dtype=np.int64)
    row_need = counts - allocation.sum(axis=1)
    col_need = class_counts - allocation.sum(axis=0)
    n_clients, n_classes = allocation.shape
    extras = np.zeros_like(allocation, dtype=bool)
    row_order = rng.permutation(n_clients)
    col_order = rng.permutation(n_classes)
    preferences = [col_order[np.argsort(-remainder[row, col_order], kind="stable")]
                   for row in range(n_clients)]

    for start_row in row_order:
        for _ in range(int(row_need[start_row])):
            row_parent = np.full(n_clients, -1, dtype=np.int64)
            col_parent = np.full(n_classes, -1, dtype=np.int64)
            visited_rows = np.zeros(n_clients, dtype=bool)
            visited_rows[start_row] = True
            queue, terminal = deque([int(start_row)]), None
            while queue and terminal is None:
                row = queue.popleft()
                for col in preferences[row]:
                    if remainder[row, col] == 0 or extras[row, col] or col_parent[col] >= 0:
                        continue
                    col_parent[col] = row
                    if col_need[col] > 0:
                        terminal = int(col)
                        break
                    # A full class column may free a slot by moving one of
                    # its previously assigned clients to another class.
                    for owner in row_order:
                        if extras[owner, col] and not visited_rows[owner]:
                            visited_rows[owner] = True
                            row_parent[owner] = col
                            queue.append(int(owner))
            if terminal is None:
                raise RuntimeError("no feasible proportional integer allocation found")
            col_need[terminal] -= 1
            col = terminal
            while col >= 0:
                row = int(col_parent[col])
                extras[row, col] = True
                previous_col = int(row_parent[row])
                if previous_col >= 0:
                    extras[row, previous_col] = False
                col = previous_col

    allocation += extras
    if not np.array_equal(allocation.sum(1), counts) or not np.array_equal(allocation.sum(0), class_counts):
        raise RuntimeError("proportional allocation did not preserve its exact marginals")
    return allocation


def partition_quantity_skew(labels, *, client_counts=None, client_fractions=None,
                            seed=0, assignment_seed=None):
    """Return ``{client_id: [example_index, ...]}`` for one training dataset.

    Supply exactly one of exact positive integer ``client_counts`` summing to
    the dataset length, or positive ``client_fractions`` (relative shares).
    IDs are consecutive integers beginning at zero. Without assignment_seed,
    quantity slots follow input order. With it, quantity slots are independently
    shuffled before assigning examples. The labels array itself is not changed.

    Apply this only to the experiment's designated training set. Create any
    validation holdout beforehand and keep its original-index mapping outside
    this module; the returned indices address precisely the provided labels.
    """
    labels = _labels(labels)
    seed = _seed(seed, "seed")
    if (client_counts is None) == (client_fractions is None):
        raise ValueError("supply exactly one of client_counts or client_fractions")
    counts = (_counts(client_counts, len(labels)) if client_counts is not None else
              allocate_client_counts(len(labels), client_fractions, seed=seed))
    if assignment_seed is not None:
        assignment_seed = _seed(assignment_seed, "assignment_seed")
        counts = counts[np.random.default_rng(assignment_seed).permutation(len(counts))]
    _, inverse, class_counts = np.unique(labels, return_inverse=True, return_counts=True)
    rng = np.random.default_rng(seed)
    allocation = _stratified_counts(counts, class_counts, rng)
    partitions = {cid: [] for cid in range(len(counts))}
    for column in range(len(class_counts)):
        indices = rng.permutation(np.flatnonzero(inverse == column))
        start = 0
        for cid in range(len(counts)):
            stop = start + int(allocation[cid, column])
            partitions[cid].extend(indices[start:stop].tolist())
            start = stop
    for indices in partitions.values():
        rng.shuffle(indices)
    return partitions


def _digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def partition_manifest(labels, partitions, *, seed=None, assignment_seed=None, dataset_id=None):
    """Validate shards and return JSON-safe counts, histograms, and hashes.

    Includes the full index lists so a result can be independently audited.
    The partition hash excludes annotation metadata and hashes sorted membership
    per client. A separate order hash detects changes to the returned local
    index order; label sequence and length are separately fingerprinted.
    """
    labels = _labels(labels)
    if not isinstance(partitions, Mapping) or not partitions:
        raise ValueError("partitions must be a nonempty mapping of client IDs to indices")
    if any(isinstance(cid, (bool, np.bool_)) or not isinstance(cid, numbers.Integral)
           for cid in partitions):
        raise ValueError("partitions must use integer client IDs")
    ids = sorted(map(int, partitions))
    if ids != list(range(len(ids))):
        raise ValueError("partitions must use consecutive client IDs starting at zero")
    if seed is not None:
        seed = _seed(seed, "seed")
    if assignment_seed is not None:
        assignment_seed = _seed(assignment_seed, "assignment_seed")
    if dataset_id is not None and not isinstance(dataset_id, str):
        raise ValueError("dataset_id must be a string or None")
    classes, inverse, global_histogram = np.unique(labels, return_inverse=True, return_counts=True)
    seen = np.zeros(len(labels), dtype=bool)
    clients, memberships, ordered = {}, {}, {}
    global_proportions = global_histogram / len(labels)
    max_quota_deviation, proportional = 0.0, True
    for cid in ids:
        raw = np.asarray(partitions[cid])
        if raw.ndim != 1 or not raw.size or raw.dtype.kind not in "iu":
            raise ValueError("partitions must contain nonempty one-dimensional integer index lists")
        if np.any(raw < 0) or np.any(raw >= len(labels)):
            raise ValueError("partitions contain an out-of-range index")
        indices = raw.astype(np.int64)
        if len(np.unique(indices)) != len(indices) or seen[indices].any():
            raise ValueError("partitions contain overlapping or duplicate indices")
        seen[indices] = True
        histogram = np.bincount(inverse[indices], minlength=len(classes))
        products = len(indices) * global_histogram.astype(object)
        lower = np.asarray(products // len(labels), dtype=np.int64)
        upper = lower + np.asarray(products % len(labels) != 0, dtype=np.int64)
        proportional = proportional and bool(np.all((histogram >= lower) & (histogram <= upper)))
        quota_deviation = float(np.max(np.abs(histogram - len(indices) * global_proportions)))
        max_quota_deviation = max(max_quota_deviation, quota_deviation)
        proportions = histogram / len(indices)
        memberships[str(cid)] = sorted(indices.tolist())
        ordered[str(cid)] = indices.tolist()
        clients[str(cid)] = {
            "n_samples": len(indices), "sample_weight": len(indices) / len(labels),
            "class_counts": histogram.tolist(), "class_proportions": proportions.tolist(),
            "classes_observed": int(np.count_nonzero(histogram)),
            "class_total_variation": float(0.5 * np.abs(proportions - global_proportions).sum()),
            "max_class_quota_deviation": quota_deviation,
            "indices": indices.tolist(), "membership_sha256": _digest(memberships[str(cid)]),
        }
    if not seen.all():
        raise ValueError("partitions do not exhaustively cover the provided labels")
    return {
        "schema_version": 1,
        "partition_kind": "quantity_only_class_stratified" if proportional else "classification_partition",
        "proportional_class_allocation": proportional,
        "validation_algorithm": "exact_ownership_and_integer_quota_check_v1",
        "dataset_id": dataset_id, "seed": seed, "assignment_seed": assignment_seed,
        "assignment_note": "Quantity assignment, hardware caps, and graph placement require independent experiment seeds.",
        "index_scope": "Indices address only the provided labels; validation/test exclusion is the caller's responsibility.",
        "n_examples": len(labels), "n_clients": len(ids),
        "class_labels": classes.tolist(), "global_class_counts": global_histogram.tolist(),
        "global_class_proportions": global_proportions.tolist(),
        "client_counts": [clients[str(cid)]["n_samples"] for cid in ids],
        "max_class_quota_deviation": max_quota_deviation,
        "coverage": {"unique_examples": int(seen.sum()), "missing_examples": 0, "duplicate_examples": 0},
        "label_sha256": _digest(labels.tolist()),
        "partition_sha256": _digest(memberships), "index_order_sha256": _digest(ordered),
        "clients": clients,
    }


__all__ = ["allocate_client_counts", "partition_quantity_skew", "partition_manifest"]
