"""Explicit Q/V LoRA for the independent Dec-LoRA comparison.

No PEFT version-dependent classifier handling: the frozen RoBERTa backbone has
Q/V adapters and a trained classification head. Both are transmitted and counted.
The paper omits head/scaling/dropout details; these are benchmark assumptions.
"""

import math
from collections.abc import Mapping
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F


def _positive_rank(rank):
    if isinstance(rank, bool) or not isinstance(rank, Integral) or rank < 1:
        raise ValueError("rank must be a positive integer")
    return int(rank)


def _positive_alpha(alpha):
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    return alpha


def _finite_tensor(value, name):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating point tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _state_schema(state):
    if not isinstance(state, Mapping) or "adapter" not in state or "head" not in state:
        raise ValueError("state must contain adapter and head mappings")
    if not isinstance(state["adapter"], Mapping) or not isinstance(state["head"], Mapping):
        raise ValueError("state must contain adapter and head mappings")


def _factor_pair(pair, name):
    if not isinstance(pair, Mapping) or "A" not in pair or "B" not in pair:
        raise ValueError(f"invalid adapter factors for {name}")
    a, b = pair["A"], pair["B"]
    _finite_tensor(a, f"{name}.A")
    _finite_tensor(b, f"{name}.B")
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1] or a.shape[0] < 1:
        raise ValueError(f"invalid adapter shape or nonpositive rank for {name}")
    if a.shape[1] < 1 or b.shape[0] < 1:
        raise ValueError(f"invalid feature shape for {name}")
    if a.dtype != b.dtype or a.device != b.device:
        raise ValueError(f"adapter factors for {name} must have matching dtype and device")
    return a, b


class QueryValueLoRA(nn.Module):
    def __init__(self, base, rank, alpha=16.0, dropout=0.0):
        super().__init__()
        rank = _positive_rank(rank)
        alpha = _positive_alpha(alpha)
        if not isinstance(base, nn.Linear):
            raise ValueError("base must be an unwrapped torch.nn.Linear")
        dropout = float(dropout)
        if not math.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError("dropout must be finite and between zero and one")
        self.base = base
        self.base.requires_grad_(False)
        self.alpha = float(alpha)
        self.dropout = nn.Dropout(dropout)
        self.A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.normal_(self.A, std=0.02)

    @property
    def rank(self):
        return self.A.shape[0]

    def forward(self, inputs):
        return self.base(inputs) + F.linear(F.linear(self.dropout(inputs), self.A), self.B) * (self.alpha / self.rank)


def inject_query_value(model, rank=16, alpha=16.0, dropout=0.0):
    rank, alpha = _positive_rank(rank), _positive_alpha(alpha)
    if not math.isfinite(float(dropout)) or not 0 <= float(dropout) <= 1:
        raise ValueError("dropout must be finite and between zero and one")
    if any(not isinstance(getattr(block.attention.self, name), nn.Linear)
           for block in model.roberta.encoder.layer for name in ("query", "value")):
        raise ValueError("query/value modules must be unwrapped linear layers; already injected?")
    model.requires_grad_(False)
    for block in model.roberta.encoder.layer:
        for name in ("query", "value"):
            attention = block.attention.self
            setattr(attention, name, QueryValueLoRA(getattr(attention, name), rank, alpha, dropout))
    model.classifier.requires_grad_(True)
    return model


def adapter_modules(model):
    return {name: module for name, module in model.named_modules() if isinstance(module, QueryValueLoRA)}


def export_state(model):
    return {
        "adapter": {name: {"A": module.A.detach().cpu().clone(), "B": module.B.detach().cpu().clone()}
                    for name, module in adapter_modules(model).items()},
        "head": {name: value.detach().cpu().clone() for name, value in model.classifier.state_dict().items()},
    }


def install_state(model, state):
    """Validate the entire state before replacing parameters or the head.

    Installing replaces adapter Parameter objects. Callers must rebuild an
    optimizer afterwards, or explicitly migrate its state to the new objects.
    """
    _state_schema(state)
    modules = adapter_modules(model)
    if set(state["adapter"]) != set(modules):
        raise ValueError("adapter layer sets differ")
    prepared = {}
    for name, module in modules.items():
        a, b = _factor_pair(state["adapter"][name], name)
        if b.shape[0] != module.base.out_features or a.shape[1] != module.base.in_features:
            raise ValueError(f"invalid adapter shape for {name}")
        device, dtype = module.base.weight.device, module.base.weight.dtype
        prepared[name] = {"A": a.to(device=device, dtype=dtype).clone(),
                          "B": b.to(device=device, dtype=dtype).clone()}
        for key, value in prepared[name].items():
            _finite_tensor(value, f"converted {name}.{key}")
    expected_head = model.classifier.state_dict()
    if set(state["head"]) != set(expected_head):
        raise ValueError("classifier head keys differ")
    head = {}
    for name, expected in expected_head.items():
        value = state["head"][name]
        _finite_tensor(value, f"head.{name}")
        if value.shape != expected.shape:
            raise ValueError(f"invalid classifier head shape for {name}")
        head[name] = value.to(device=expected.device, dtype=expected.dtype).clone()
        _finite_tensor(head[name], f"converted head.{name}")
    for name, module in modules.items():
        module.A = nn.Parameter(prepared[name]["A"])
        module.B = nn.Parameter(prepared[name]["B"])
    model.classifier.load_state_dict(head)


def expand_state(state, target_rank, alpha=16.0, generator=None):
    """Increase rank without changing the update; give new zero-B rows live A.

    Simultaneously zero A/B padding would make the new coordinates untrainable.
    Existing factors are rescaled because the forward convention is alpha/r.
    This helper does not truncate or shrink (use an explicit projection there).
    """
    target_rank = _positive_rank(target_rank)
    _positive_alpha(alpha)
    _state_schema(state)
    for name, value in state["head"].items():
        _finite_tensor(value, f"head.{name}")
    result = {"adapter": {}, "head": {k: v.clone() for k, v in state["head"].items()}}
    for name, pair in state["adapter"].items():
        a, b = _factor_pair(pair, name)
        rank = a.shape[0]
        if target_rank < rank:
            raise ValueError("expand_state cannot shrink")
        scale = math.sqrt(target_rank / rank)
        if target_rank == rank:
            result["adapter"][name] = {"A": a.clone(), "B": b.clone()}
            continue
        extra_a = torch.randn(target_rank - rank, a.shape[1], dtype=a.dtype, device=a.device, generator=generator) * 0.02
        result["adapter"][name] = {
            "A": torch.cat((a * scale, extra_a), dim=0),
            "B": torch.cat((b * scale, torch.zeros(b.shape[0], target_rank - rank, dtype=b.dtype, device=b.device)), dim=1),
        }
        _factor_pair(result["adapter"][name], name)
    return result


def state_bytes(state):
    adapter = sum(v.numel() * v.element_size() for pair in state["adapter"].values() for v in pair.values())
    head = sum(v.numel() * v.element_size() for v in state["head"].values())
    return {"adapter": adapter, "head": head, "total": adapter + head}
