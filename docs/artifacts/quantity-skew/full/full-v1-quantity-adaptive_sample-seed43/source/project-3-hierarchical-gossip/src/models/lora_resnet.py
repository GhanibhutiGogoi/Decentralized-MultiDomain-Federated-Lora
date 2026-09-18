"""
ResNet-18 with LoRA adapters for federated learning experiments.

Uses a custom LoRA implementation on the final FC layer since PEFT's
LoRA is designed for transformer models. For ResNet, we manually add
low-rank adapters to linear layers.
"""

import copy
import math
import torch
import torch.nn as nn
from torchvision import models


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA (Low-Rank Adaptation).

    Wraps an existing linear layer, freezes its weights, and adds
    trainable low-rank matrices A and B such that:
        output = frozen_linear(x) + (x @ A^T @ B^T) * (alpha / rank)
    """

    def __init__(self, original_linear, rank=16, alpha=32):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze original weights
        self.linear = original_linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

        # Initialize A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_output = self.linear(x)
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_output + lora_output

    def get_lora_params(self):
        """Return LoRA A and B matrices as detached CPU tensors."""
        return {
            'A': self.lora_A.data.detach().cpu().clone(),
            'B': self.lora_B.data.detach().cpu().clone(),
        }

    def set_lora_params(self, params):
        """Set LoRA A and B matrices from dict."""
        self.lora_A.data.copy_(params['A'])
        self.lora_B.data.copy_(params['B'])


def create_lora_resnet(
    num_classes=100,
    rank=16,
    alpha=32,
    pretrained=True,
    device='cpu',
):
    """
    Create a ResNet-18 model with LoRA on the final FC layer.

    Args:
        num_classes: number of output classes
        rank: LoRA rank
        alpha: LoRA scaling factor
        pretrained: whether to use ImageNet pretrained weights
        device: target device

    Returns:
        model: ResNet-18 with LoRA
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Replace the final FC layer with LoRA version
    original_fc = model.fc
    # First adjust for new number of classes if needed
    if original_fc.out_features != num_classes:
        original_fc = nn.Linear(original_fc.in_features, num_classes)

    model.fc = LoRALinear(original_fc, rank=rank, alpha=alpha)

    # Freeze all base model parameters
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False

    model = model.to(device)
    return model


def get_lora_state(model):
    """
    Extract all LoRA parameters from model.

    Returns:
        dict mapping layer_name -> {'A': tensor, 'B': tensor}
    """
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[name] = module.get_lora_params()
    return lora_state


def set_lora_state(model, lora_state):
    """Set all LoRA parameters in model from state dict."""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and name in lora_state:
            module.set_lora_params(lora_state[name])


def get_trainable_param_count(model):
    """Count trainable vs total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def clone_model(model, device='cpu'):
    """Create a deep copy of the model."""
    cloned = copy.deepcopy(model)
    return cloned.to(device)
