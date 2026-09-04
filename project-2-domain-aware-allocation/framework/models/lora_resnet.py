"""
ResNet-18 with LoRA adapters supporting variable ranks.

Extends the base LoRA implementation with utilities for:
- Merging LoRA to rank-independent delta_W (for heterogeneous aggregation)
- SVD decomposition back to target rank
- Resetting LoRA parameters for oracle search
"""

import copy
import math
import torch
import torch.nn as nn
from torchvision import models


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA (Low-Rank Adaptation).

    output = frozen_linear(x) + (x @ A^T @ B^T) * (alpha / rank)
    """

    def __init__(self, original_linear, rank=16, alpha=32):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.linear = original_linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

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
        rank: LoRA rank (can be any value in {4, 8, 16, 32, 64})
        alpha: LoRA scaling factor
        pretrained: whether to use ImageNet pretrained weights
        device: target device

    Returns:
        model: ResNet-18 with LoRA
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    original_fc = model.fc
    if original_fc.out_features != num_classes:
        original_fc = nn.Linear(original_fc.in_features, num_classes)

    model.fc = LoRALinear(original_fc, rank=rank, alpha=alpha)

    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False

    model = model.to(device)
    return model


def get_lora_state(model):
    """Extract all LoRA parameters from model."""
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


def reset_lora_params(model):
    """
    Re-initialize all LoRA parameters to their initial values.
    A: Kaiming uniform, B: zeros. This makes LoRA start as identity again.
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
            nn.init.zeros_(module.lora_B)


def merge_lora_to_delta_w(lora_state):
    """
    Compute delta_W = B @ A for each LoRA layer.

    This produces a rank-independent representation: the effective weight
    update is always (out_features x in_features) regardless of the rank
    used to produce it.

    Args:
        lora_state: dict mapping layer_name -> {'A': tensor, 'B': tensor}

    Returns:
        dict mapping layer_name -> delta_W tensor (out_features x in_features)
    """
    delta_w = {}
    for layer_name, params in lora_state.items():
        A = params['A']  # (rank, in_features)
        B = params['B']  # (out_features, rank)
        delta_w[layer_name] = B @ A  # (out_features, in_features)
    return delta_w


def decompose_delta_w(delta_w_dict, target_rank, alpha=32):
    """
    SVD-decompose delta_W back into A, B matrices at target_rank.

    Given delta_W = U @ S @ V^T, we take the top-r singular values:
        B_new = U[:, :r] @ sqrt(S[:r])  -> (out_features, r)
        A_new = sqrt(S[:r]) @ V^T[:r, :] -> (r, in_features)

    Args:
        delta_w_dict: dict mapping layer_name -> delta_W tensor
        target_rank: desired rank for the decomposition
        alpha: LoRA alpha for scaling

    Returns:
        lora_state dict compatible with set_lora_state
    """
    lora_state = {}
    for layer_name, delta_w in delta_w_dict.items():
        U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)

        r = min(target_rank, len(S))
        sqrt_s = torch.sqrt(S[:r])

        B_new = U[:, :r] * sqrt_s.unsqueeze(0)   # (out, r)
        A_new = sqrt_s.unsqueeze(1) * Vh[:r, :]   # (r, in)

        # Scale by 1/scaling since LoRA forward multiplies by alpha/rank
        scaling = alpha / target_rank
        B_new = B_new / math.sqrt(scaling)
        A_new = A_new / math.sqrt(scaling)

        lora_state[layer_name] = {
            'A': A_new,
            'B': B_new,
        }
    return lora_state
