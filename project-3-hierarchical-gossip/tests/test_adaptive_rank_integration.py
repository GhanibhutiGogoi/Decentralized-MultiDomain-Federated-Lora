"""Check the actual P1 policy bridge and trainable, leak-free rank changes."""

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.protocol_benchmark import FeatureClient
from src.federated import adaptive_rank
from src.federated.merge import lora_to_delta


def _client(rank=4):
    rng = torch.Generator().manual_seed(91)
    initial = {
        "weight": torch.randn(5, 12, generator=rng),
        "bias": torch.randn(5, generator=rng),
        "A": torch.randn(16, 12, generator=rng) * 0.1,
    }
    config = SimpleNamespace(alpha=16.0, current_seed=42, batch_size=8,
                             eval_batch_size=16, lr=0.001,
                             weight_decay=0.0, local_epochs=1)
    train = {"features": torch.randn(40, 12, generator=rng),
             "labels": torch.randint(0, 5, (40,), generator=rng)}
    # Invalid held-out labels make accidental probe leakage fail loudly.
    test = {"features": torch.randn(10, 12, generator=rng),
            "labels": torch.full((10,), -999)}
    return FeatureClient(0, 0, initial, rank, config, train, test,
                         {"train_indices": list(range(3, 35)),
                          "test_indices": list(range(10))}, torch.device("cpu"))


def test_controller_definitions_are_verbatim_canonical_p1_source():
    repo = Path(__file__).resolve().parents[2]
    canonical = repo / adaptive_rank.POLICY_PROVENANCE["source"]
    config = repo / adaptive_rank.POLICY_PROVENANCE["config_source"]
    assert hashlib.sha256(canonical.read_bytes()).hexdigest() == adaptive_rank.POLICY_PROVENANCE["source_sha256"]
    assert hashlib.sha256(config.read_bytes()).hexdigest() == adaptive_rank.POLICY_PROVENANCE["config_source_sha256"]
    names = {"AdaptiveRankController", "_nearest_candidate", "_ceil_candidate",
             "capability_fraction", "rank_equation"}
    def definitions(path):
        return {node.name: ast.dump(node, include_attributes=False)
                for node in ast.parse(path.read_text()).body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names}
    assert definitions(canonical) == definitions(Path(adaptive_rank.__file__))


@pytest.mark.parametrize("ceiling", [4, 8, 16])
def test_published_policy_has_warmup_floor_and_quality_recovery(ceiling):
    controller = adaptive_rank.make_controller(ceiling)
    assert [controller.update(1.0) for _ in range(2)] == [ceiling, ceiling]
    ranks = [controller.update(1.0) for _ in range(20)]
    assert all(ceiling // 2 <= rank <= ceiling for rank in ranks)
    assert ranks[-1] == ceiling // 2
    controller.observe_quality(0.8)
    controller.observe_quality(0.6)
    assert controller.update(1.0) == ceiling


@pytest.mark.parametrize("unsupported", [0, 2, 12, 32])
def test_policy_rejects_unpublished_capability_ceiling(unsupported):
    with pytest.raises(ValueError, match="rank ceilings"):
        adaptive_rank.make_controller(unsupported)


def test_expansion_preserves_delta_and_new_directions_have_gradients():
    client = _client(rank=2)
    with torch.no_grad():
        client.model.lora_B.copy_(torch.arange(10).reshape(5, 2) * 0.01)
    before = lora_to_delta(client.get_lora_state(), client.alpha)["fc"]
    base_weight = client.model.linear.weight.detach().clone()
    global_rng = torch.random.get_rng_state().clone()
    train_rng = client.rng.get_state().clone()
    client.resize_rank(4)
    after = lora_to_delta(client.get_lora_state(), client.alpha)["fc"]
    torch.testing.assert_close(after, before, atol=1e-6, rtol=1e-6)
    assert torch.equal(client.model.linear.weight, base_weight)
    assert torch.equal(torch.random.get_rng_state(), global_rng)
    assert torch.equal(client.rng.get_state(), train_rng)
    assert bool(torch.all(client.model.lora_A[2:].abs().sum(1) > 0))
    assert not bool(client.model.lora_B[:, 2:].any())
    index = client.probe_indices
    loss = torch.nn.functional.cross_entropy(
        client.model(client.train_data["features"][index]), client.train_data["labels"][index])
    loss.backward()
    assert bool(torch.all(client.model.lora_B.grad[:, 2:].abs().sum(0) > 0))


def test_shrink_is_best_truncated_effective_update():
    client = _client(rank=4)
    rng = torch.Generator().manual_seed(4)
    with torch.no_grad():
        client.model.lora_B.copy_(torch.randn(5, 4, generator=rng))
    before = lora_to_delta(client.get_lora_state(), client.alpha)["fc"]
    singular = torch.linalg.svdvals(before)
    client.resize_rank(2)
    after = lora_to_delta(client.get_lora_state(), client.alpha)["fc"]
    torch.testing.assert_close(torch.sum((before - after) ** 2),
                               torch.sum(singular[2:] ** 2), rtol=1e-5, atol=1e-6)


def test_zero_update_shrink_remains_trainable():
    client = _client(rank=4)
    client.resize_rank(2)
    assert not bool(lora_to_delta(client.get_lora_state(), client.alpha)["fc"].any())
    assert bool(torch.all(client.model.lora_A.abs().sum(1) > 0))
    client.train()
    assert bool(client.model.lora_B.any())


def test_probe_uses_only_train_data_without_mutating_model_or_rng():
    client = _client(rank=2)
    client.model.lora_A.grad = torch.full_like(client.model.lora_A, 7.0)
    state = {name: value.clone() for name, value in client.model.state_dict().items()}
    global_rng = torch.random.get_rng_state().clone()
    train_rng = client.rng.get_state().clone()
    rank_rng = client.rank_rng.get_state().clone()
    probe_indices = client.probe_indices.clone()
    first = client.probe_gradient_stable_rank(4)
    second = client.probe_gradient_stable_rank(4)
    quality = client.probe_quality()
    assert first == second
    assert 1 - 1e-5 <= first <= 4 + 1e-5
    assert 0 < quality < 1
    assert torch.equal(torch.random.get_rng_state(), global_rng)
    assert torch.equal(client.rng.get_state(), train_rng)
    assert torch.equal(client.rank_rng.get_state(), rank_rng)
    assert torch.equal(client.probe_indices, probe_indices)
    assert set(probe_indices.tolist()).issubset(set(client.train_indices.tolist()))
    assert len(probe_indices) == client.config.batch_size
    assert bool(torch.all(client.model.lora_A.grad == 7.0))
    assert client.rank == 2
    for name, value in client.model.state_dict().items():
        assert torch.equal(value, state[name])


def test_probe_rejects_ceiling_below_active_rank():
    with pytest.raises(ValueError, match="current rank"):
        _client(rank=4).probe_gradient_stable_rank(2)
