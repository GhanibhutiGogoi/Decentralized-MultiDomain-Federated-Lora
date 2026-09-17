"""Independent factor/effective merge oracles and serialized graph accounting."""

import copy

import numpy as np
import pytest
import torch

from src.federated.quantity_transport import factor_mix, mix_heads, neighbor_round, tree_assemble


def state(rank=2, value=1.0, dtype=torch.float64):
    a = (torch.arange(rank * 5, dtype=dtype).reshape(rank, 5) + value) / 10
    b = (torch.arange(4 * rank, dtype=dtype).reshape(4, rank) - value) / 7
    return {"adapter": {"q": {"A": a, "B": b}},
            "head": {"weight": torch.full((3, 4), value, dtype=dtype),
                     "bias": torch.tensor([value, -value, 2 * value], dtype=dtype),
                     "scalar": torch.tensor(value, dtype=dtype)}}


def ring(n):
    return {i: sorted({(i - 1) % n, (i + 1) % n} - {i}) for i in range(n)}


def effective(peer, alpha):
    factors = peer["adapter"]["q"]
    return (alpha / factors["A"].shape[0]) * factors["B"] @ factors["A"]


def tensor_bytes(peer):
    values = [tensor for factors in peer["adapter"].values() for tensor in factors.values()]
    values += list(peer["head"].values())
    return sum(t.numel() * t.element_size() for t in values)


def assert_same(left, right):
    for name, factors in left["adapter"].items():
        for factor, tensor in factors.items():
            torch.testing.assert_close(tensor, right["adapter"][name][factor], atol=1e-12, rtol=1e-12)
    for name, tensor in left["head"].items():
        torch.testing.assert_close(tensor, right["head"][name], atol=1e-12, rtol=1e-12)


def test_factor_baseline_and_head_use_the_same_convex_weights():
    peers = [state(value=1), state(value=4)]
    original = copy.deepcopy(peers)
    merged = factor_mix(peers, [1, 3])
    for factor in ("A", "B"):
        expected = .25 * peers[0]["adapter"]["q"][factor] + .75 * peers[1]["adapter"]["q"][factor]
        torch.testing.assert_close(merged["adapter"]["q"][factor], expected)
    for name in peers[0]["head"]:
        torch.testing.assert_close(merged["head"][name], .25 * peers[0]["head"][name] + .75 * peers[1]["head"][name])
    assert not torch.allclose(effective(merged, 16), .25 * effective(peers[0], 16) + .75 * effective(peers[1], 16))
    for before, after in zip(original, peers):
        assert_same(before, after)
    assert_same(factor_mix(peers, [1e300, 3e300]), merged)


def test_factor_baseline_rejects_heterogeneous_ranks_but_head_mix_allows_them():
    peers = [state(rank=1, value=1), state(rank=3, value=3)]
    with pytest.raises(ValueError, match="equal ranks"):
        factor_mix(peers, [1, 1])
    torch.testing.assert_close(mix_heads(peers, [1, 1])["bias"], torch.tensor([2., -2., 4.], dtype=torch.float64))


def test_ring_tree_assembly_counts_each_source_once_per_hop_and_disseminates():
    peers = {cid: state(value=cid + 1) for cid in range(6)}
    weights = {cid: cid + 1 for cid in peers}
    graph = ring(6)
    assembled, ledger = tree_assemble(peers, weights, graph, root_id=0, mode="factor", target_rank=2)
    assert_same(assembled, factor_mix(list(peers.values()), list(weights.values())))
    expected_depth = {0: 0, 1: 1, 2: 2, 3: 3, 4: 2, 5: 1}
    assert ledger["tree_depth"] == expected_depth
    assert ledger["gather_tensor_bytes"] == sum(expected_depth[cid] * tensor_bytes(peer) for cid, peer in peers.items())
    assert ledger["dissemination_tensor_bytes"] == 5 * tensor_bytes(assembled)
    assert ledger["gather_messages"] == ledger["dissemination_messages"] == 5
    assert ledger["messages"] == 10
    assert ledger["bytes"] == ledger["tensor_bytes"] + ledger["metadata_bytes"]
    assert ledger["metadata_bytes"] > 0
    assert ledger["root_assembly_input_tensor_bytes"] == sum(tensor_bytes(peer) for peer in peers.values())
    assert ledger["peak_gather_record_tensor_bytes_by_peer"][0] == ledger["root_assembly_input_tensor_bytes"]
    for row in ledger["transfers"]:
        assert row["receiver"] in graph[row["sender"]]
        assert row["payload_bytes"] == row["tensor_bytes"] + row["metadata_bytes"]
        assert len(row["payload_sha256"]) == 64
    broadcasts = [row for row in ledger["transfers"] if row["phase"] == "dissemination"]
    assert len({row["payload_sha256"] for row in broadcasts}) == 1
    assert ledger["normalized_weights"] == pytest.approx({cid: (cid + 1) / 21 for cid in peers})


def test_effective_tree_assembly_matches_independent_dense_product_and_head_oracles():
    peers = {0: state(rank=1, value=1), 1: state(rank=2, value=4), 2: state(rank=3, value=2)}
    weights = {0: 1, 1: 2, 2: 3}
    assembled, ledger = tree_assemble(peers, weights, ring(3), mode="effective", target_rank=4, alpha=16, disseminate=False)
    expected = sum(weights[cid] / 6 * effective(peer, 16) for cid, peer in peers.items())
    torch.testing.assert_close(effective(assembled, 16), expected, atol=1e-11, rtol=1e-11)
    for name in peers[0]["head"]:
        torch.testing.assert_close(assembled["head"][name], sum(weights[cid] / 6 * peer["head"][name] for cid, peer in peers.items()))
    assert ledger["dissemination_bytes"] == 0
    assert ledger["gather_messages"] == 2
    assert ledger["merge_diagnostics"]["relative_discarded_energy"] < 1e-20


def test_single_peer_has_no_invented_network_traffic():
    peer = state()
    assembled, ledger = tree_assemble({0: peer}, {0: 3}, {0: []}, mode="factor", target_rank=2)
    assert_same(assembled, peer)
    assert ledger["messages"] == ledger["bytes"] == ledger["max_wire_payload_bytes"] == 0


def test_synchronous_neighbor_round_uses_only_old_graph_neighbor_states():
    peers = {cid: state(value=cid + 1) for cid in range(4)}
    original = copy.deepcopy(peers)
    graph = ring(4)
    matrix = np.array([[.5, .25, 0, .25], [.2, .6, .2, 0], [0, .3, .4, .3], [.1, 0, .1, .8]])
    merged, ledger = neighbor_round(peers, matrix, graph, {cid: 2 for cid in peers}, 16, "factor")
    for cid in peers:
        assert_same(merged[cid], factor_mix(list(original.values()), matrix[cid]))
        assert_same(peers[cid], original[cid])
    assert ledger["gossip_messages"] == 8
    assert ledger["gossip_tensor_bytes"] == 2 * sum(tensor_bytes(peer) for peer in peers.values())
    assert ledger["gather_bytes"] == ledger["dissemination_bytes"] == 0
    for row in ledger["transfers"]:
        assert row["receiver"] in graph[row["sender"]]
        assert row["source_weights"] == [matrix[row["receiver"], row["sender"]]]


def test_heterogeneous_neighbor_compact_merge_matches_dense_truncation():
    peers = {0: state(rank=1, value=1), 1: state(rank=2, value=2), 2: state(rank=3, value=5)}
    matrix = np.array([[.5, .25, .25], [.2, .5, .3], [.1, .2, .7]])
    ranks = {0: 1, 1: 2, 2: 3}
    merged, _ = neighbor_round(peers, matrix, ring(3), ranks, 16, "effective")
    for cid in peers:
        dense = sum(matrix[cid, sender] * effective(peers[sender], 16) for sender in peers)
        u, s, vh = torch.linalg.svd(dense, full_matrices=False)
        expected = (u[:, :ranks[cid]] * s[:ranks[cid]]) @ vh[:ranks[cid]]
        torch.testing.assert_close(effective(merged[cid], 16), expected, atol=1e-10, rtol=1e-10)
        assert merged[cid]["adapter"]["q"]["A"].shape[0] == ranks[cid]


def test_non_neighbor_contributions_and_disconnected_assembly_are_rejected():
    peers = {cid: state() for cid in range(4)}
    with pytest.raises(ValueError, match="non-neighbor"):
        neighbor_round(peers, np.full((4, 4), .25), ring(4), {cid: 2 for cid in peers}, 16, "factor")
    with pytest.raises(ValueError, match="connected"):
        tree_assemble(peers, {cid: 1 for cid in peers}, {0: [1], 1: [0], 2: [3], 3: [2]}, mode="factor")


@pytest.mark.parametrize("weights", [[0, 0], [-1, 2], [1, float("nan")], [1], [[1, 2]]])
def test_invalid_weights_fail(weights):
    with pytest.raises(ValueError, match="weights"):
        factor_mix([state(), state()], weights)


def test_invalid_state_and_rank_changes_are_rejected():
    bad = state()
    bad["head"]["bias"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        factor_mix([state(), bad], [1, 1])
    with pytest.raises(ValueError, match="cannot change"):
        tree_assemble({0: state()}, {0: 1}, {0: []}, mode="factor", target_rank=3)
    with pytest.raises(ValueError, match="positive integer"):
        tree_assemble({0: state()}, {0: 1}, {0: []}, mode="effective", target_rank=0)
