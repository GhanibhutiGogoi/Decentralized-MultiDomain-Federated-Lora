"""
Federated CIFAR-100 Non-IID Data Pipeline
=========================================

What this code does:
This script provides a complete, standalone PyTorch DataLoader pipeline designed 
for Federated Learning experiments. It takes the standard CIFAR-100 dataset and 
partitions it into distinct client datasets using a Dirichlet distribution to 
simulate real-world, non-IID (highly skewed) data distributions.

Datasets & Structure:
- Base Dataset: CIFAR-100 (60,000 images, 32x32 RGB).
- Classes: 100 "fine" classes grouped into 20 "coarse" superclasses.
- Domains: The code treats the 20 coarse superclasses as distinct "domains" 
  (e.g., aquatic mammals, flowers, vehicles).
- Custom Wrapper (CIFAR100DomainDataset): Modifies the standard dataset to yield a 
  tuple of (image, fine_label, domain_label) for every sample. This is specifically 
  designed for Domain Generalization (DG) or Federated Domain Adaptation tasks.

Non-IID Partitioning:
- Uses a Dirichlet distribution (parameterized by `alpha`) over the labels.
- Alpha (α): Controls the severity of the data skew. 
    - α = 0.1  -> Pathological non-IID (clients get mostly 1-2 classes/domains).
    - α = 0.5  -> Moderate non-IID.
    - α = 10.0 -> Near-IID (data is relatively evenly distributed across clients).

Outputs:
- client_train_loaders: A list of PyTorch DataLoaders, one for each federated client.
- global_test_loader: A centralized PyTorch DataLoader for global model evaluation.
- client_indices: A dictionary mapping each client to their specific data indices.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np

# ---------------------------------------------------------
# 1. CIFAR-100 Fine-to-Coarse (Domain) Mapping
# ---------------------------------------------------------
# CIFAR-100 has 100 classes mapped to 20 superclasses (domains).
# Index represents the fine label (0-99), value represents the domain (0-19).
FINE_TO_COARSE = np.array([
    4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
    6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
    5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 16,  4, 17,  4,  2,  0, 17,  4, 18, 17,
   10,  3,  2, 12, 12, 16, 12,  1,  9, 19,  2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
   16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 18,  1,  2, 15,  6,  0, 17,  8, 14, 13
])

DOMAIN_NAMES = [
    "aquatic mammals", "fish", "flowers", "food containers", "fruit and vegetables",
    "household electrical devices", "household furniture", "insects", "large carnivores",
    "large man-made outdoor things", "large natural outdoor scenes", "large omnivores and herbivores",
    "medium-sized mammals", "non-insect invertebrates", "people", "reptiles", "small mammals",
    "trees", "vehicles 1", "vehicles 2"
]

class CIFAR100DomainDataset(Dataset):
    """Wraps CIFAR-100 to yield (image, fine_label, domain_label)."""
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        img, fine_label = self.base_dataset[idx]
        domain_label = FINE_TO_COARSE[fine_label]
        return img, fine_label, domain_label

# ---------------------------------------------------------
# 2. Non-IID Dirichlet Partitioning
# ---------------------------------------------------------
def partition_dirichlet_non_iid(dataset, num_clients, alpha=0.5, num_classes=100):
    """
    Partitions dataset indices using a Dirichlet distribution over labels.
    Lower alpha -> higher non-IID (more skewed distribution per client).
    """
    # Extract labels from the base CIFAR-100 dataset
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'base_dataset'):
        labels = np.array(dataset.base_dataset.targets)
    else:
        raise ValueError("Cannot extract targets from dataset.")

    client_indices = {i: [] for i in range(num_clients)}
    
    # Iterate over each class to distribute its samples among clients
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        
        # Draw proportions from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Balance proportions (prevent clients from getting 0 samples if alpha is extremely low)
        proportions = np.array([p * (len(idx_j) < len(labels) / num_clients) for p, idx_j in zip(proportions, client_indices.values())])
        proportions = proportions / proportions.sum()
        
        # Split indices based on proportions
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        splits = np.split(idx_k, proportions)
        
        for i in range(num_clients):
            client_indices[i].extend(splits[i].tolist())
            
    # Shuffle indices within each client
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
        
    return client_indices

# ---------------------------------------------------------
# 3. Loader Generation Pipeline
# ---------------------------------------------------------
def get_federated_cifar100_loaders(num_clients=5, alpha=0.1, batch_size=64):
    """
    Downloads, wraps, partitions, and returns train/test DataLoaders.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    # Load Base Datasets
    base_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    base_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    # Wrap with Domain Labels
    domain_train = CIFAR100DomainDataset(base_train)
    domain_test = CIFAR100DomainDataset(base_test)

    # Partition Training Data (Non-IID based on fine labels)
    print(f"Partitioning {len(base_train)} training samples into {num_clients} clients (Dirichlet α={alpha})...")
    client_indices = partition_dirichlet_non_iid(domain_train, num_clients, alpha=alpha, num_classes=100)

    # Create DataLoaders
    client_train_loaders = []
    for i in range(num_clients):
        subset = Subset(domain_train, client_indices[i])
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_train_loaders.append(loader)

    # Global Test Loader
    global_test_loader = DataLoader(domain_test, batch_size=batch_size, shuffle=False)

    return client_train_loaders, global_test_loader, client_indices

# ---------------------------------------------------------
# 4. Verification & Usage Example
# ---------------------------------------------------------
if __name__ == "__main__":
    NUM_CLIENTS = 4
    ALPHA = 0.5  # Try 0.1 for extreme non-IID, or 10.0 for near-IID
    
    loaders, test_loader, indices_dict = get_federated_cifar100_loaders(
        num_clients=NUM_CLIENTS, 
        alpha=ALPHA, 
        batch_size=128
    )
    
    # ---------------------------------------------------------
    # Analyze the Non-IID Distribution across Domains
    # ---------------------------------------------------------
    print("\n--- Domain Distribution per Client ---")
    
    # Collect domain distributions to prove it's non-IID
    client_domain_counts = np.zeros((NUM_CLIENTS, 20))
    
    for client_id, loader in enumerate(loaders):
        total_samples = 0
        for _, _, domains in loader: # Unpacking the 3 variables: img, fine, domain
            for d in domains:
                client_domain_counts[client_id, d.item()] += 1
            total_samples += len(domains)
            
        print(f"Client {client_id}: {total_samples} samples.")
        
        # Print top 3 dominant domains for this client
        top_domains = np.argsort(client_domain_counts[client_id])[-3:][::-1]
        print(f"  Top Domains:")
        for td in top_domains:
            count = int(client_domain_counts[client_id, td])
            pct = (count / total_samples) * 100 if total_samples > 0 else 0
            print(f"    - {DOMAIN_NAMES[td]}: {count} ({pct:.1f}%)")
            
    # ---------------------------------------------------------
    # Ready-to-use Training Loop Snippet
    # ---------------------------------------------------------
    """
    for client_idx, train_loader in enumerate(loaders):
        # Initialize model for client
        for epoch in range(local_epochs):
            for batch_idx, (images, fine_labels, domain_labels) in enumerate(train_loader):
                images, fine_labels, domain_labels = images.to(device), fine_labels.to(device), domain_labels.to(device)
                
                # If doing standard classification:
                # outputs = model(images)
                # loss = criterion(outputs, fine_labels)
                
                # If doing Domain Generalization/Adaptation:
                # outputs = model(images, domain_labels) ... 
    """
