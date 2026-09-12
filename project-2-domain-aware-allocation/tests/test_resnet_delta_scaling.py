"""Effective-weight transport must match LoRALinear's alpha/r forward scale."""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

PROJECT2_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT2_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT2_ROOT))

from framework.federated.server import HeteroFedAvgServer
from framework.models.lora_resnet import (
    LoRALinear,
    decompose_delta_w,
    get_lora_state,
    merge_lora_to_delta_w,
    set_lora_state,
)


@pytest.mark.parametrize("target_rank", [2, 4, 8])
def test_transport_preserves_forward_outputs_across_rank_and_alpha(target_rank):
    generator = torch.Generator().manual_seed(81)
    source = LoRALinear(nn.Linear(5, 3), rank=2, alpha=12)
    with torch.no_grad():
        source.lora_A.copy_(torch.randn(2, 5, generator=generator))
        source.lora_B.copy_(torch.randn(3, 2, generator=generator))

    destination = LoRALinear(nn.Linear(5, 3), rank=target_rank, alpha=7)
    destination.linear.load_state_dict(source.linear.state_dict())
    update = merge_lora_to_delta_w(get_lora_state(source), alpha=source.alpha)
    set_lora_state(destination, decompose_delta_w(update, target_rank, alpha=destination.alpha))

    x = torch.randn(11, 5, generator=generator)
    torch.testing.assert_close(destination(x), source(x), atol=1e-4, rtol=1e-5)
    assert destination.lora_A.shape == (target_rank, 5)
    assert destination.lora_B.shape == (3, target_rank)


def test_compression_retains_the_optimal_effective_update():
    update = torch.diag(torch.tensor([9.0, 4.0, 1.0], dtype=torch.float64))
    state = decompose_delta_w({"fc": update}, target_rank=2, alpha=5)
    reconstructed = merge_lora_to_delta_w(state, alpha=5)["fc"]
    torch.testing.assert_close(reconstructed, torch.diag(torch.tensor([9.0, 4.0, 0.0], dtype=torch.float64)))


class _NoTrainingClient:
    """Isolate communication from training in an actual server round."""

    def __init__(self, client_id, rank, alpha, value, samples):
        self.client_id = client_id
        self.domain_id = 0
        self.samples = samples
        self.state = decompose_delta_w({"fc": torch.diag(torch.tensor([value, 0.0]))}, rank, alpha)

    def train(self):
        return {"n_samples": self.samples}

    def get_lora_state(self):
        return self.state

    def set_lora_state(self, state):
        self.state = state

    def evaluate(self):
        return {"accuracy": 0.0}


def test_heterogeneous_server_averages_effective_updates_without_roundwise_shrinkage():
    alpha = 10
    clients = [
        _NoTrainingClient(0, rank=1, alpha=alpha, value=2.0, samples=1),
        _NoTrainingClient(1, rank=2, alpha=alpha, value=6.0, samples=3),
    ]
    server = HeteroFedAvgServer(clients, {0: 1, 1: 2}, alpha=alpha)
    server.run(n_rounds=3, verbose=False)
    expected = torch.diag(torch.tensor([5.0, 0.0]))
    for client in clients:
        reconstructed = merge_lora_to_delta_w(client.state, alpha=alpha)["fc"]
        torch.testing.assert_close(reconstructed, expected)


@pytest.mark.parametrize("alpha", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_scaling_is_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        decompose_delta_w({"fc": torch.eye(3)}, target_rank=2, alpha=alpha)
    with pytest.raises(ValueError, match="alpha"):
        merge_lora_to_delta_w({}, alpha=alpha)
