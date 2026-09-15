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


def merge_lora_to_delta_w(lora_state, alpha=32):
    """
    Compute the effective delta_W = (alpha / rank) * B @ A per layer.

    This produces a rank-independent representation: the effective weight
    update is always (out_features x in_features) regardless of the rank
    used to produce it.

    Args:
        lora_state: dict mapping layer_name -> {'A': tensor, 'B': tensor}
        alpha: the LoRA alpha used by the source model (shared across layers)

    Returns:
        dict mapping layer_name -> delta_W tensor (out_features x in_features)
    """
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    delta_w = {}
    for layer_name, params in lora_state.items():
        A = params['A']  # (rank, in_features)
        B = params['B']  # (out_features, rank)
        if A.ndim != 2 or B.ndim != 2 or A.shape[0] <= 0 or B.shape[1] != A.shape[0]:
            raise ValueError(f"inconsistent LoRA factors for {layer_name!r}")
        if not torch.is_floating_point(A) or not torch.is_floating_point(B):
            raise ValueError(f"LoRA factors for {layer_name!r} must be floating-point tensors")
        if not torch.isfinite(A).all() or not torch.isfinite(B).all():
            raise ValueError(f"LoRA factors for {layer_name!r} must contain only finite values")
        delta_w[layer_name] = (alpha / A.shape[0]) * (B @ A)
    return delta_w


def decompose_delta_w(delta_w_dict, target_rank, alpha=32):
    """
    SVD-decompose an effective delta_W into factors at target_rank.

    Given delta_W = U @ S @ V^T, we take the top-r singular values:
        B_new = U[:, :r] @ sqrt(S[:r])  -> (out_features, r)
        A_new = sqrt(S[:r]) @ V^T[:r, :] -> (r, in_features)

    Args:
        delta_w_dict: dict mapping layer_name -> delta_W tensor
        target_rank: desired rank for the decomposition
        alpha: destination model's LoRA alpha; its forward scaling is undone
            in the factors, so merging them recovers the effective delta_W

    Returns:
        lora_state dict compatible with set_lora_state
    """
    if target_rank <= 0:
        raise ValueError("target_rank must be positive")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    lora_state = {}
    for layer_name, delta_w in delta_w_dict.items():
        if delta_w.ndim != 2:
            raise ValueError(f"delta_W for {layer_name!r} must be 2-D")
        if delta_w.numel() == 0:
            raise ValueError(f"delta_W for {layer_name!r} must be non-empty")
        if not torch.is_floating_point(delta_w):
            raise ValueError(f"delta_W for {layer_name!r} must be a floating-point tensor")
        if not torch.isfinite(delta_w).all():
            raise ValueError(f"delta_W for {layer_name!r} must contain only finite values")
        # CPU SVD does not implement half/bfloat16, but preserve double input
        # precision and restore the original dtype before returning.
        work = delta_w if delta_w.dtype == torch.float64 else delta_w.float()
        U, S, Vh = torch.linalg.svd(work, full_matrices=False)

        r = min(target_rank, len(S))
        sqrt_s = torch.sqrt(S[:r])

        B_new = U[:, :r] * sqrt_s.unsqueeze(0)   # (out, r)
        A_new = sqrt_s.unsqueeze(1) * Vh[:r, :]   # (r, in)

        # Scale by 1/scaling since LoRA forward multiplies by alpha/rank
        scaling = alpha / target_rank
        B_new = B_new / math.sqrt(scaling)
        A_new = A_new / math.sqrt(scaling)

        # The requested LoRA rank may exceed the layer's intrinsic dimension.
        # Still return tensors that can be loaded into the destination model.
        if r < target_rank:
            B_new = torch.cat([B_new, B_new.new_zeros(B_new.shape[0], target_rank - r)], dim=1)
            A_new = torch.cat([A_new, A_new.new_zeros(target_rank - r, A_new.shape[1])], dim=0)

        lora_state[layer_name] = {
            'A': A_new.to(delta_w.dtype),
            'B': B_new.to(delta_w.dtype),
        }
    return lora_state
