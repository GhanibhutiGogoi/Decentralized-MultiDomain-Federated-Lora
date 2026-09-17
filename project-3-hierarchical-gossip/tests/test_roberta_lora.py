"""Architecture-level LoRA checks without downloading a model or dataset.

Execute only on the authorized remote test host. A tiny randomly initialized
RoBERTa exercises the real attention/head integration instead of mocking it.
"""

import copy

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import RobertaConfig, RobertaForSequenceClassification

from src.models.roberta_lora import (
    QueryValueLoRA,
    adapter_modules,
    expand_state,
    export_state,
    inject_query_value,
    install_state,
    state_bytes,
)


def tiny_model(seed=314):
    torch.manual_seed(seed)
    config = RobertaConfig(
        vocab_size=97,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=40,
        max_position_embeddings=40,
        type_vocab_size=1,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout=0.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        num_labels=2,
    )
    config._attn_implementation = "eager"
    return RobertaForSequenceClassification(config)


def inputs():
    ids = torch.tensor([[0, 4, 7, 9, 2, 1], [0, 3, 8, 10, 11, 2], [0, 6, 5, 2, 1, 1]])
    return {"input_ids": ids, "attention_mask": ids.ne(1).long(), "labels": torch.tensor([0, 1, 0])}


def make_adapter_nonzero(model, seed=2718):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in adapter_modules(model).values():
            module.B.copy_(torch.randn(module.B.shape, generator=generator) * .03)


def test_zero_adapter_update_preserves_full_model_predictions_exactly():
    model = tiny_model().eval()
    with torch.no_grad():
        baseline = model(**inputs()).logits.clone()
    inject_query_value(model, rank=4, alpha=16, dropout=0)
    assert len(adapter_modules(model)) == 4  # Q and V at each of two layers.
    assert all(torch.count_nonzero(module.B) == 0 for module in adapter_modules(model).values())
    with torch.no_grad():
        adapted = model(**inputs()).logits
    torch.testing.assert_close(adapted, baseline, atol=0, rtol=0)


def test_only_adapter_and_classifier_are_trainable_and_backbone_does_not_change():
    model = inject_query_value(tiny_model(), rank=4, alpha=16)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    expected = {f"{name}.{factor}" for name in adapter_modules(model) for factor in ("A", "B")}
    expected |= {f"classifier.{name}" for name, _ in model.classifier.named_parameters()}
    assert trainable == expected
    frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
              if not parameter.requires_grad}
    trainable_before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                        if parameter.requires_grad}
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                  lr=.01, weight_decay=0)
    model.train()
    model(**inputs()).loss.backward()
    assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
    assert all(module.B.grad is not None and torch.count_nonzero(module.B.grad) > 0
               for module in adapter_modules(model).values())
    assert model.classifier.out_proj.weight.grad is not None
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name in frozen:
            assert torch.equal(parameter, frozen[name]), name
    current = dict(model.named_parameters())
    assert any(not torch.equal(current[name], value) for name, value in trainable_before.items()
               if name.startswith("classifier."))
    assert any(not torch.equal(current[name], value) for name, value in trainable_before.items()
               if name.endswith(".B"))


def test_export_install_roundtrip_reproduces_logits_and_has_no_aliases():
    model = inject_query_value(tiny_model(), rank=3, alpha=16).eval()
    replica = copy.deepcopy(model)
    make_adapter_nonzero(model)
    with torch.no_grad():
        model.classifier.out_proj.bias.add_(torch.tensor([.1, -.1]))
        expected = model(**inputs()).logits
    exported = export_state(model)
    install_state(replica, exported)
    with torch.no_grad():
        actual = replica(**inputs()).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert all(value.device.type == "cpu" for pair in exported["adapter"].values() for value in pair.values())
    first = next(iter(exported["adapter"]))
    installed_a = adapter_modules(replica)[first].A.detach().clone()
    original_a = adapter_modules(model)[first].A.detach().clone()
    exported["adapter"][first]["A"].zero_()
    assert torch.equal(adapter_modules(replica)[first].A, installed_a)
    assert torch.equal(adapter_modules(model)[first].A, original_a)
    count = state_bytes(export_state(model))
    expected_count = sum(parameter.numel() * parameter.element_size()
                         for parameter in model.parameters() if parameter.requires_grad)
    assert count["total"] == count["adapter"] + count["head"] == expected_count


def test_expand_preserves_effective_updates_logits_and_new_b_coordinates_can_learn():
    model = inject_query_value(tiny_model(), rank=2, alpha=16).eval()
    make_adapter_nonzero(model)
    original = export_state(model)
    with torch.no_grad():
        before = model(**inputs()).logits.clone()
    expanded = expand_state(original, 6, alpha=16, generator=torch.Generator().manual_seed(123))
    for name, pair in original["adapter"].items():
        larger = expanded["adapter"][name]
        torch.testing.assert_close((16 / 6) * (larger["B"] @ larger["A"]),
                                   (16 / 2) * (pair["B"] @ pair["A"]), atol=2e-8, rtol=2e-6)
        assert torch.count_nonzero(larger["A"][2:]) > 0
        assert torch.count_nonzero(larger["B"][:, 2:]) == 0
    install_state(model, expanded)
    with torch.no_grad():
        after = model(**inputs()).logits
    torch.testing.assert_close(after, before, atol=2e-8, rtol=2e-6)
    model.zero_grad(set_to_none=True)
    model(**inputs()).loss.backward()
    for name, module in adapter_modules(model).items():
        assert module.rank == 6
        assert module.B.grad is not None, name
        assert torch.count_nonzero(module.B.grad[:, 2:]) > 0, name
    assert all(torch.equal(expanded["head"][name], value) for name, value in original["head"].items())


def test_expand_same_rank_clones_and_shrinking_is_explicitly_rejected():
    model = inject_query_value(tiny_model(), rank=3, alpha=16)
    original = export_state(model)
    cloned = expand_state(original, 3, alpha=16)
    for name, pair in original["adapter"].items():
        for key, value in pair.items():
            assert torch.equal(cloned["adapter"][name][key], value)
            assert cloned["adapter"][name][key].data_ptr() != value.data_ptr()
    with pytest.raises(ValueError, match="shrink"):
        expand_state(original, 2, alpha=16)


def test_rank_heterogeneous_clients_share_backbone_head_and_zero_effective_initialization():
    common = inject_query_value(tiny_model(), rank=8, alpha=16).eval()
    common_state = export_state(common)
    peers = []
    for rank in (2, 4, 8):
        peer = copy.deepcopy(common)
        sliced = {"adapter": {name: {"A": pair["A"][:rank].clone(), "B": pair["B"][:, :rank].clone()}
                               for name, pair in common_state["adapter"].items()},
                  "head": {name: value.clone() for name, value in common_state["head"].items()}}
        install_state(peer, sliced)
        peers.append(peer)
        for name, parameter in peer.named_parameters():
            if not parameter.requires_grad:
                assert torch.equal(parameter, dict(common.named_parameters())[name]), name
        for name, module in adapter_modules(peer).items():
            assert module.rank == rank
            assert torch.equal(module.A, common_state["adapter"][name]["A"][:rank])
    with torch.no_grad():
        logits = [peer(**inputs()).logits for peer in peers]
    for value in logits[1:]:
        torch.testing.assert_close(value, logits[0], atol=0, rtol=0)


def test_install_rejects_wrong_layer_set_and_factor_shape():
    model = inject_query_value(tiny_model(), rank=2, alpha=16)
    state = export_state(model)
    missing = copy.deepcopy(state)
    del missing["adapter"][next(iter(missing["adapter"]))]
    with pytest.raises(ValueError, match="layer sets"):
        install_state(model, missing)
    malformed = copy.deepcopy(state)
    first = next(iter(malformed["adapter"]))
    malformed["adapter"][first]["B"] = torch.zeros(24, 3)
    with pytest.raises(ValueError, match="shape"):
        install_state(model, malformed)


@pytest.mark.parametrize("rank", [0, -1, 1.5, True])
def test_invalid_construction_and_expansion_ranks_rejected(rank):
    with pytest.raises(ValueError, match="rank"):
        QueryValueLoRA(torch.nn.Linear(5, 7), rank)
    original = export_state(inject_query_value(tiny_model(), rank=2))
    with pytest.raises(ValueError, match="rank"):
        expand_state(original, rank)


@pytest.mark.parametrize("fault", ["rank_zero", "nan_factor", "inf_head", "head_shape", "head_keys", "overflow_cast"])
def test_invalid_install_is_rejected_before_any_parameter_changes(fault):
    model = inject_query_value(tiny_model(), rank=2)
    before = export_state(model)
    candidate = copy.deepcopy(before)
    names = list(candidate["adapter"])
    # A valid early layer differs, so a late failure catches partial mutation.
    candidate["adapter"][names[0]]["A"].add_(1)
    last = candidate["adapter"][names[-1]]
    if fault == "rank_zero":
        last["A"], last["B"] = torch.empty(0, 24), torch.empty(24, 0)
    elif fault == "nan_factor":
        last["B"][0, 0] = float("nan")
    elif fault == "inf_head":
        candidate["head"]["out_proj.bias"][0] = float("inf")
    elif fault == "head_shape":
        candidate["head"]["out_proj.bias"] = torch.zeros(3)
    elif fault == "head_keys":
        del candidate["head"]["out_proj.bias"]
    elif fault == "overflow_cast":
        last["A"] = last["A"].double().fill_(1e100)
        last["B"] = last["B"].double()
    with pytest.raises(ValueError):
        install_state(model, candidate)
    after = export_state(model)
    for name, pair in before["adapter"].items():
        for key, value in pair.items():
            assert torch.equal(after["adapter"][name][key], value)
    for name, value in before["head"].items():
        assert torch.equal(after["head"][name], value)


@pytest.mark.parametrize("alpha", [0, -1, float("nan"), float("inf")])
def test_invalid_alpha_rejected_at_construction_and_expansion(alpha):
    with pytest.raises(ValueError, match="alpha"):
        QueryValueLoRA(torch.nn.Linear(5, 7), 2, alpha=alpha)
    original = export_state(inject_query_value(tiny_model(), rank=2))
    with pytest.raises(ValueError, match="alpha"):
        expand_state(original, 4, alpha=alpha)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_adapter_initialization_inherits_base_device_and_dtype(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    base = torch.nn.Linear(5, 7).to(device=device, dtype=torch.float64)
    module = QueryValueLoRA(base, 3)
    assert module.A.device == module.B.device == base.weight.device
    assert module.A.dtype == module.B.dtype == torch.float64
    example = torch.randn(2, 5, device=device, dtype=torch.float64)
    torch.testing.assert_close(module(example), base(example), atol=0, rtol=0)


def test_second_injection_rejected_without_freezing_existing_adapters():
    model = inject_query_value(tiny_model(), rank=2)
    with pytest.raises(ValueError, match="already injected"):
        inject_query_value(model, rank=4)
    assert all(module.A.requires_grad and module.B.requires_grad for module in adapter_modules(model).values())
