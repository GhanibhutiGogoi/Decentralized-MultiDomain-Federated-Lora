"""
CIFAR-100 domain splitting and non-IID client partitioning.

Splits CIFAR-100 into 5 domains based on superclasses, then distributes
data across clients using Dirichlet distribution for non-IID partitioning.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# CIFAR-100 has 20 superclasses, each with 5 fine classes (100 total).
# We group superclasses into 5 domains of 4 superclasses each (20 classes per domain).
# Mapping: superclass index -> fine class indices (from CIFAR-100 metadata)
SUPERCLASS_TO_FINE = {
    0: [4, 30, 55, 72, 95],      # aquatic mammals
    1: [1, 32, 67, 73, 91],      # fish
    2: [54, 62, 70, 82, 92],     # flowers
    3: [9, 10, 16, 28, 61],      # food containers
    4: [0, 51, 53, 57, 83],      # fruit and vegetables
    5: [22, 39, 40, 86, 87],     # household electrical devices
    6: [5, 20, 25, 84, 94],      # household furniture
    7: [6, 7, 14, 18, 24],       # insects
    8: [3, 42, 43, 88, 97],      # large carnivores
    9: [12, 17, 37, 68, 76],     # large man-made outdoor things
    10: [23, 33, 49, 60, 71],    # large natural outdoor scenes
    11: [15, 19, 21, 31, 38],    # large omnivores and herbivores
    12: [34, 63, 64, 66, 75],    # medium-sized mammals
    13: [26, 45, 77, 79, 99],    # non-insect invertebrates
    14: [2, 11, 35, 46, 98],     # people
    15: [27, 29, 44, 78, 93],    # reptiles
    16: [36, 50, 65, 74, 80],    # small mammals
    17: [47, 52, 56, 59, 96],    # trees
    18: [8, 13, 48, 58, 90],     # vehicles 1
    19: [41, 69, 81, 85, 89],    # vehicles 2
}

# 5 domains: each domain = 4 superclasses = 20 fine classes
DOMAIN_SUPERCLASSES = {
    0: [0, 1, 2, 3],     # aquatic + fish + flowers + food
    1: [4, 5, 6, 7],     # fruit/veg + household + insects
    2: [8, 9, 10, 11],   # large animals + outdoor
    3: [12, 13, 14, 15],  # medium mammals + invertebrates + people + reptiles
    4: [16, 17, 18, 19],  # small mammals + trees + vehicles
}

DOMAIN_NAMES = {
    0: "Aquatic & Nature",
    1: "Household & Insects",
    2: "Large Animals & Outdoor",
    3: "Medium Animals & People",
    4: "Small Animals & Vehicles",
}


def get_domain_classes(domain_id):
    """Get the list of fine-grained class indices for a domain."""
    superclasses = DOMAIN_SUPERCLASSES[domain_id]
    classes = []
    for sc in superclasses:
        classes.extend(SUPERCLASS_TO_FINE[sc])
    return sorted(classes)


def get_transforms(train=True):
    """Standard CIFAR-100 transforms."""
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            ),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            ),
        ])


def partition_domain_data_dirichlet(indices, targets, n_clients, alpha=0.5, seed=42):
    """
    Partition data indices among n_clients using Dirichlet distribution.

    Args:
        indices: array of dataset indices belonging to this domain
        targets: array of class labels for the full dataset
        n_clients: number of clients to split across
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        seed: random seed

    Returns:
        List of index arrays, one per client
    """
    rng = np.random.default_rng(seed)
    domain_targets = targets[indices]
    unique_classes = np.unique(domain_targets)

    client_indices = [[] for _ in range(n_clients)]

    for cls in unique_classes:
        cls_mask = domain_targets == cls
        cls_indices = indices[cls_mask]
        rng.shuffle(cls_indices)

        # Dirichlet distribution over clients for this class
        proportions = rng.dirichlet(np.ones(n_clients) * alpha)
        # Ensure at least 1 sample per client if possible
        proportions = proportions / proportions.sum()
        split_points = (np.cumsum(proportions) * len(cls_indices)).astype(int)
        split_points[-1] = len(cls_indices)

        splits = np.split(cls_indices, split_points[:-1])
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    # Shuffle each client's data
    for i in range(n_clients):
        rng.shuffle(client_indices[i])

    return client_indices


def create_federated_datasets(
    data_dir="./data",
    n_domains=5,
    clients_per_domain=3,
    dirichlet_alpha=0.5,
    batch_size=64,
    seed=42,
):
    """
    Create federated dataset splits for CIFAR-100.

    Returns:
        clients_data: dict mapping client_id -> {
            'train_loader': DataLoader,
            'test_loader': DataLoader,
            'domain_id': int,
            'domain_name': str,
            'n_samples': int,
            'classes': list of class indices
        }
        test_dataset: full test dataset for global evaluation
    """
    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True,
        transform=get_transforms(train=True)
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True,
        transform=get_transforms(train=False)
    )

    train_targets = np.array(train_dataset.targets)
    test_targets = np.array(test_dataset.targets)

    clients_data = {}
    client_id = 0

    for domain_id in range(n_domains):
        domain_classes = get_domain_classes(domain_id)

        # Get indices of samples belonging to this domain
        train_mask = np.isin(train_targets, domain_classes)
        test_mask = np.isin(test_targets, domain_classes)
        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]

        # Partition training data among clients using Dirichlet
        client_train_splits = partition_domain_data_dirichlet(
            train_indices, train_targets, clients_per_domain,
            alpha=dirichlet_alpha, seed=seed + domain_id
        )

        # Test data: split evenly among clients in this domain
        test_per_client = np.array_split(test_indices, clients_per_domain)

        for local_id in range(clients_per_domain):
            train_subset = Subset(train_dataset, client_train_splits[local_id])
            test_subset = Subset(test_dataset, test_per_client[local_id].tolist())

            train_loader = DataLoader(
                train_subset, batch_size=batch_size, shuffle=True,
                num_workers=2, pin_memory=True
            )
            test_loader = DataLoader(
                test_subset, batch_size=batch_size, shuffle=False,
                num_workers=2, pin_memory=True
            )

            clients_data[client_id] = {
                'train_loader': train_loader,
                'test_loader': test_loader,
                'domain_id': domain_id,
                'domain_name': DOMAIN_NAMES[domain_id],
                'n_samples': len(client_train_splits[local_id]),
                'classes': domain_classes,
            }
            client_id += 1

    return clients_data, test_dataset


def print_data_summary(clients_data):
    """Print summary of data distribution across clients."""
    print("\n" + "=" * 60)
    print("Federated Data Distribution Summary")
    print("=" * 60)

    for cid, info in clients_data.items():
        print(f"  Client {cid:2d} | Domain {info['domain_id']} "
              f"({info['domain_name']:30s}) | "
              f"Samples: {info['n_samples']:5d}")

    print("=" * 60)
    total = sum(info['n_samples'] for info in clients_data.values())
    print(f"  Total samples: {total}")
    print(f"  Total clients: {len(clients_data)}")
    print(f"  Domains: {len(set(info['domain_id'] for info in clients_data.values()))}")
    print("=" * 60 + "\n")
