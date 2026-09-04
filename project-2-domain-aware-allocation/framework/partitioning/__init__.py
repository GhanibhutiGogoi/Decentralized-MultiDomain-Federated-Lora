"""Partitioning utilities."""

from .dirichlet import dirichlet_label_partition_indices
from .iid import iid_partition_indices
from .labels import extract_labels, num_classes_from_labels
from .partitioning import (
    PartitionConfig,
    make_client_loaders,
    make_client_subsets,
    partition_indices,
)
from .statistics import (
    class_imbalance_ratio,
    client_class_counts,
    entropy,
    normalized,
)

__all__ = [
    "PartitionConfig",
    "class_imbalance_ratio",
    "client_class_counts",
    "dirichlet_label_partition_indices",
    "entropy",
    "extract_labels",
    "iid_partition_indices",
    "make_client_loaders",
    "make_client_subsets",
    "normalized",
    "num_classes_from_labels",
    "partition_indices",
]
