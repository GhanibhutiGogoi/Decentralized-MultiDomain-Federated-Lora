"""Focused regression tests for reviewer-requested input validation."""
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.aggregation.fedavg import _normalised_client_weights, fedavg_quality_weighted
from framework.aggregation.projection import project_tensor_to_rank
from framework.federated.server import HeteroFedAvgServer


def test_normalised_weights_reject_malformed_inputs():
    with pytest.raises(ValueError, match="at least one"):
        _normalised_client_weights([], [])
    with pytest.raises(ValueError, match="matching lengths"):
        _normalised_client_weights([1], [1, 2])
    with pytest.raises(ValueError, match="nonnegative"):
        _normalised_client_weights([1, -1], [1, 1])
    with pytest.raises(ValueError, match="finite"):
        _normalised_client_weights([float("nan")], [1])
    with pytest.raises(ValueError, match="positive total"):
        _normalised_client_weights([0], [1])


def _server():
    return HeteroFedAvgServer([], {}, alpha=4)


@pytest.mark.parametrize(
    "updates, weights, pattern",
    [([], [], "at least one"), ([{"x": torch.ones(2, 2)}], [], "matching lengths"),
     ([{"x": torch.ones(2, 2)}], [-1], "nonnegative"),
     ([{"x": torch.ones(2, 2)}], [0], "positive total"),
     ([{"x": torch.ones(2, 2)}, {"y": torch.ones(2, 2)}], [1, 1], "layer keys"),
     ([{"x": torch.ones(2, 2)}, {"x": torch.ones(3, 2)}], [1, 1], "shape")],
)
def test_delta_aggregation_rejects_invalid_inputs(updates, weights, pattern):
    with pytest.raises(ValueError, match=pattern):
        _server().aggregate_delta_w(updates, weights)


def test_projection_rejects_nonpositive_rank():
    tensor = torch.ones(2, 3)
    with pytest.raises(ValueError, match="positive integer"):
        project_tensor_to_rank(tensor, 0)
    with pytest.raises(ValueError, match="positive integer"):
        project_tensor_to_rank(tensor, -2)


class _AlphaClient:
    def __init__(self, alpha):
        self.client_id = alpha
        self.domain_id = 0
        self.model = nn.Module()
        self.model.alpha = alpha


def test_server_rejects_mixed_client_alpha():
    clients = [_AlphaClient(4), _AlphaClient(8)]
    with pytest.raises(ValueError, match="alpha"):
        HeteroFedAvgServer(clients, {4: 1, 8: 1}, alpha=4)


def test_server_accepts_shared_client_alpha():
    clients = [_AlphaClient(4), _AlphaClient(4)]
    HeteroFedAvgServer(clients, {4: 1}, alpha=4)


@pytest.mark.parametrize("states, samples, quality", [([], [], []), ([{}], [1, 2], [1, 1]), ([{}, {}], [1], [1])])
def test_quality_fedavg_rejects_missing_or_unmatched_client_states(states, samples, quality):
    with pytest.raises(ValueError, match="at least one|matching lengths"):
        fedavg_quality_weighted(states, samples, quality, 2, {}, "cpu")


def test_normalised_weights_do_not_overflow_at_float32_range():
    assert _normalised_client_weights([1e100, 1e100], [1, 1]) == [0.5, 0.5]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_aggregation_rejects_nonfinite_weights(invalid):
    with pytest.raises(ValueError, match="finite"):
        _server().aggregate_delta_w([{"x": torch.ones(2, 2)}], [invalid])
    with pytest.raises(ValueError, match="finite"):
        _normalised_client_weights([1], [invalid])


def test_aggregation_rejects_nontensor_reference():
    with pytest.raises(ValueError, match="tensor"):
        _server().aggregate_delta_w([{"x": 1.0}, {"x": torch.ones(2, 2)}], [1, 1])


def test_server_rejects_missing_alpha_metadata():
    with pytest.raises(ValueError, match="expose a shared alpha"):
        HeteroFedAvgServer([object()], {}, alpha=4)


def test_server_revalidates_alpha_before_training():
    client = _AlphaClient(4)
    server = HeteroFedAvgServer([client], {4: 1}, alpha=4)
    client.model.alpha = 8
    with pytest.raises(ValueError, match="alpha"):
        server.run(n_rounds=1, verbose=False)


def test_normalised_weights_preserve_zero_weight_clients():
    assert _normalised_client_weights([0, 3], [2, 4]) == [0.0, 1.0]


def test_delta_aggregation_uses_normalized_weights():
    actual = _server().aggregate_delta_w(
        [{"x": torch.full((2, 2), 2.0)}, {"x": torch.full((2, 2), 6.0)}], [1, 3]
    )
    torch.testing.assert_close(actual["x"], torch.full((2, 2), 5.0))
