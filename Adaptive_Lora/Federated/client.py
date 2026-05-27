# client-side training and quality scoring

import torch
import torch.nn as nn
import torch.optim as optim


def train_client(model, loader, epochs, device):
    """
    Train model on local data for a given number of epochs.
    Returns (state_dict, total_samples_seen).
    """
    model.train()
    opt     = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    total   = 0

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
            total += y.size(0)

    return model.state_dict(), total


def compute_quality_score(model, loader, loss_fn, device, num_batches=5):
    """
    Quality score = 1 / (1 + avg_train_loss).
    Higher score = lower loss = update is more reliable.
    A client forced into too-high rank will have higher loss → lower score.
    """
    model.eval()
    total_loss = 0.0
    count      = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= num_batches:
                break
            x, y = x.to(device), y.to(device)
            total_loss += loss_fn(model(x), y).item()
            count      += 1

    avg_loss = total_loss / max(count, 1)
    return 1.0 / (1.0 + avg_loss)
