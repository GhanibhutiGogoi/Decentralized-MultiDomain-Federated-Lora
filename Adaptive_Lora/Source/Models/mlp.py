# MLP and TabularMLP model definitions

import torch
import torch.nn as nn
from .lora import LoRALinear


class MLP(nn.Module):
    """
    Two-layer MLP for image classification (e.g. FashionMNIST).
    LoRA applied to both layers.
    """
    def __init__(self, in_dim, num_classes=10, r=4):
        super().__init__()
        self.fc1 = LoRALinear(in_dim, 128, r)
        self.fc2 = LoRALinear(128, num_classes, r)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x.view(x.size(0), -1))))


class TabularMLP(nn.Module):
    """
    Deeper MLP for tabular data classification.
    LoRA applied to all linear layers.
    """
    def __init__(self, in_dim, num_classes=2, r=4):
        super().__init__()
        self.net  = nn.Sequential(
            LoRALinear(in_dim, 256, r), nn.ReLU(), nn.Dropout(0.3),
            LoRALinear(256, 128, r),   nn.ReLU(), nn.Dropout(0.2),
            LoRALinear(128, 64, r),    nn.ReLU())
        self.head = LoRALinear(64, num_classes, r)

    def forward(self, x):
        return self.head(self.net(x))
