import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from src.clustering.discovery import OnlineDomainDiscovery, AdaptiveAffinityMixer


def _state(rows, seed=0):
    a = torch.tensor([[1.0, -2.0, 0.5], [0.5, 1.0, -1.0]])
    b = torch.zeros(6, 2)
    for i, row in enumerate(rows):
        b[row, i % 2] = 1.0
    return {"fc": {"A": a, "B": b}}


def test_online_discovery_is_label_free_and_recovers_separated_domains():
    states = [_state([0, 1], i) for i in range(3)] + [_state([4, 5], i + 20) for i in range(3)]
    d = OnlineDomainDiscovery(beta=0.0, max_clusters=4)
    snap = d.update(states, alpha=32)
    assert snap.n_clusters == 2
    assert snap.signature_dimension <= 64
    assert adjusted_rand_score([0, 0, 0, 1, 1, 1], snap.labels) == 1.0
    assert np.allclose(snap.affinity, snap.affinity.T)


def test_discovery_matrix_is_symmetric_doubly_stochastic():
    ids = list(range(6))
    states = [_state([0, 1]) for _ in range(3)] + [_state([4, 5]) for _ in range(3)]
    mixer = AdaptiveAffinityMixer(ids, alpha=32,
                                  discovery=OnlineDomainDiscovery(beta=0.0))
    mixer.update(states)
    w = mixer(0)
    assert w.shape == (6, 6)
    assert np.all(w >= 0)
    assert np.allclose(w, w.T, atol=1e-9)
    assert np.allclose(w.sum(axis=0), 1.0, atol=1e-8)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-8)


def test_ema_prevents_single_round_cluster_flip():
    states_a = [_state([0, 1]) for _ in range(2)] + [_state([4, 5]) for _ in range(2)]
    states_b = [_state([4, 5]) for _ in range(2)] + [_state([0, 1]) for _ in range(2)]
    d = OnlineDomainDiscovery(beta=0.9, max_clusters=3)
    first = d.update(states_a, 32).labels.copy()
    second = d.update(states_b, 32).labels.copy()
    assert adjusted_rand_score(first, second) > 0.0
