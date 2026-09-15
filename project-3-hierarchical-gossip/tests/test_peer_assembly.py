"""Remote-testable invariants of final neighbor-only adapter assembly."""

import pytest
import torch

from src.federated.merge import factorize_delta, lora_to_delta
from src.federated.mixing import build_topology
from src.federated.peer_assembly import tree_allgather, tree_weighted_assembly


def state(rank, seed):
    generator = torch.Generator().manual_seed(seed)
    return {"fc": {"A": torch.randn(rank, 7, generator=generator, dtype=torch.float64),
                   "B": torch.randn(5, rank, generator=generator, dtype=torch.float64)}}


def test_ring_tree_assembly_matches_scaled_heterogeneous_weighted_target():
    states = {10: state(1, 1), 20: state(3, 2), 30: state(2, 3), 40: state(4, 4)}
    weights = {10: 1.0, 20: 2.0, 30: 5.0, 40: 3.0}
    graph = build_topology([30, 10, 40, 20], "ring")
    alpha = 16.0
    assembled, info = tree_weighted_assembly(states, weights, graph, 40, 5, alpha)
    expected = sum(weights[cid] * lora_to_delta(value, alpha)["fc"]
                   for cid, value in states.items()) / sum(weights.values())
    assert torch.allclose(lora_to_delta(assembled, alpha)["fc"], expected, atol=1e-11)
    assert info["messages"] == len(states) - 1
    assert info["dense_floats"] == 3 * 5 * 7
    assert info["scalar_metadata_values"] == 3
    assert info["bytes"] == 3 * (5 * 7 * 8 + 8)
    assert info["total_weight"] == sum(weights.values())
    assert info["relative_truncation_energy"] < 1e-25
    for message in info["tree_edges"]:
        assert message["receiver"] in graph[message["sender"]]
    # Every subtree must arrive before its parent sends, with no central read
    # of the individual peer factors represented by a non-neighbor edge.
    sent = set()
    tree_parent = {item["sender"]: item["receiver"] for item in info["tree_edges"]}
    for item in info["tree_edges"]:
        children = {cid for cid, parent in tree_parent.items() if parent == item["sender"]}
        assert children <= sent
        sent.add(item["sender"])
    assert sent == set(states) - {40}


def test_truncates_only_at_deployment_peer_and_reports_loss():
    states = {0: state(2, 1), 1: state(4, 2), 2: state(3, 3)}
    weights = {0: 1.0, 1: 7.0, 2: 2.0}
    actual, info = tree_weighted_assembly(states, weights, build_topology(list(states), "path"),
                                         2, 2, 16.0)
    ideal = sum(weights[cid] * lora_to_delta(value, 16.0)["fc"]
                for cid, value in states.items()) / 10.0
    target = lora_to_delta({"fc": factorize_delta(ideal, 2, 16.0)}, 16.0)["fc"]
    observed = lora_to_delta(actual, 16.0)["fc"]
    assert torch.allclose(observed, target, atol=1e-11)
    assert info["truncation_energy"] == pytest.approx(float(((ideal - observed) ** 2).sum()))
    assert info["truncation_energy"] > 0


def test_one_peer_needs_no_transport():
    states = {7: state(2, 3)}
    result, info = tree_weighted_assembly(states, {7: 3.0}, {7: []}, 7, 2, 16.0)
    assert info["messages"] == info["bytes"] == info["floats"] == 0
    assert torch.allclose(lora_to_delta(result, 16.0)["fc"],
                          lora_to_delta(states[7], 16.0)["fc"], atol=1e-11)


@pytest.mark.parametrize("weights", [{0: 0, 1: 1}, {0: -1, 1: 2},
                                     {0: float("nan"), 1: 1}, {0: float("inf"), 1: 1}])
def test_refuses_invalid_weights(weights):
    with pytest.raises(ValueError, match="weights"):
        tree_weighted_assembly({0: state(1, 0), 1: state(2, 1)}, weights,
                                {0: [1], 1: [0]}, 0, 2, 16.0)


def test_refuses_disconnected_or_nonreciprocal_transport():
    states = {0: state(1, 0), 1: state(2, 1)}
    with pytest.raises(ValueError, match="connected"):
        tree_weighted_assembly(states, {0: 1, 1: 1}, {0: [], 1: []}, 0, 2, 16.0)
    with pytest.raises(ValueError, match="symmetric"):
        tree_weighted_assembly(states, {0: 1, 1: 1}, {0: [1], 1: []}, 0, 2, 16.0)


def test_weight_rescaling_leaves_assembled_model_unchanged():
    states = {0: state(1, 0), 1: state(2, 1)}
    graph = {0: [1], 1: [0]}
    first, _ = tree_weighted_assembly(states, {0: 2, 1: 7}, graph, 1, 4, 16.0)
    second, _ = tree_weighted_assembly(states, {0: 20, 1: 70}, graph, 1, 4, 16.0)
    assert torch.allclose(lora_to_delta(first, 16.0)["fc"],
                          lora_to_delta(second, 16.0)["fc"], atol=1e-11)


def test_numeric_allgather_routes_full_records_and_charges_every_hop():
    payloads = {0: {"n": 1, "q": 0.4, "hist": [1, 0], "delta": [0.2, 0.3]},
                1: {"n": 2, "q": 0.5, "hist": [0, 2], "delta": [0.1, 0.7]},
                2: {"n": 3, "q": 0.6, "hist": [1, 2], "delta": [0.8, 0.5]}}
    graph = build_topology([0, 1, 2], "path")
    views, info = tree_allgather(payloads, graph, root_id=0)
    for peer in payloads:
        assert views[peer] == payloads
    assert info["messages"] == 4
    # Six numbers and one source ID per record. Gather sends 1 then 2
    # records; broadcast sends all 3 records on both edges: 9 records total.
    assert info["numeric_values"] == 9 * 6
    assert info["source_identifier_values"] == 9
    assert info["bytes"] == 9 * 7 * 8
    assert info["full_view_bytes_per_peer"] == 3 * 7 * 8
    for message in info["tree_edges"]:
        assert message["receiver"] in graph[message["sender"]]
    assert [m["phase"] for m in info["tree_edges"]] == ["gather", "gather", "broadcast", "broadcast"]


def test_allgather_singleton_has_no_transmission_and_rejects_nonfinite_records():
    payloads = {5: {"delta": torch.ones(2, 3), "n": 7}}
    views, info = tree_allgather(payloads, {5: []}, 5)
    assert views[5][5] is payloads[5]
    assert info["messages"] == info["bytes"] == 0
    with pytest.raises(ValueError, match="finite"):
        tree_allgather({5: {"q": float("nan")}}, {5: []}, 5)
