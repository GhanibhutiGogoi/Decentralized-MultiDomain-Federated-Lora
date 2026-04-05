"""
Federated learning client for local training and evaluation.
"""

import torch
import torch.nn as nn
from src.models.lora_resnet import get_lora_state, set_lora_state


class FederatedClient:
    """
    A federated learning client that performs local training on private data.

    Each client holds:
    - A local copy of the model (shared base + private LoRA adapters)
    - Its own training and test data
    - Domain membership information
    """

    def __init__(
        self,
        client_id,
        model,
        train_loader,
        test_loader,
        domain_id,
        lr=0.001,
        local_epochs=5,
        device='cpu',
    ):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.domain_id = domain_id
        self.lr = lr
        self.local_epochs = local_epochs
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        # Only optimize LoRA parameters
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=1e-4,
        )

    def train(self):
        """
        Perform local training for `local_epochs` epochs.

        Returns:
            dict with training metrics (loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(self.local_epochs):
            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * batch_x.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(batch_y).sum().item()
                total += batch_y.size(0)

        avg_loss = total_loss / total if total > 0 else 0
        accuracy = correct / total if total > 0 else 0

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'n_samples': total,
        }

    @torch.no_grad()
    def evaluate(self):
        """
        Evaluate the model on local test data.

        Returns:
            dict with evaluation metrics (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in self.test_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)

        avg_loss = total_loss / total if total > 0 else 0
        accuracy = correct / total if total > 0 else 0

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'n_samples': total,
        }

    def get_lora_state(self):
        """Get current LoRA parameters."""
        return get_lora_state(self.model)

    def set_lora_state(self, state):
        """Set LoRA parameters from external state."""
        set_lora_state(self.model, state)
        # Reset optimizer state since parameters changed
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=1e-4,
        )
