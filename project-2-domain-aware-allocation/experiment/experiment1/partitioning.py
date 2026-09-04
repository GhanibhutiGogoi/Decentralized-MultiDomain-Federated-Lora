"""Compatibility re-export for the centralized partitioning framework."""

from framework.partitioning import (
    PartitionConfig,
    dirichlet_label_partition_indices,
    extract_labels,
    iid_partition_indices,
    make_client_loaders,
    make_client_subsets,
    num_classes_from_labels,
    partition_indices,
)

__all__ = [
    "PartitionConfig",
    "dirichlet_label_partition_indices",
    "extract_labels",
    "iid_partition_indices",
    "make_client_loaders",
    "make_client_subsets",
    "num_classes_from_labels",
    "partition_indices",
]

