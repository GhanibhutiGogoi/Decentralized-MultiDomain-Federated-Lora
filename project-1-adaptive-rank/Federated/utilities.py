# server-side utilities

import torch
from torch.utils.data import Subset


def split_dataset(dataset, num_clients=3):
    """Evenly split a dataset into num_clients non-overlapping subsets."""
    n    = len(dataset)
    size = n // num_clients
    return [
        Subset(dataset, list(range(i * size, (i + 1) * size)))
        for i in range(num_clients)
    ]


def evaluate(model, loader, device):
    """
    Evaluate model accuracy on a data loader.
    Returns accuracy as a percentage (0–100).
    """
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y     = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total   += y.size(0)
    return 100.0 * correct / total if total else 0.0
