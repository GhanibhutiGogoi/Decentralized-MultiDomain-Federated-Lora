"""Scientific invariants for the exploratory interventions, run on gpu003."""
import numpy as np
import torch

from experiments.diagnose_optimization import centered_state, iid_splits, matrix_diagnostics
from experiments.protocol_benchmark import make_splits
from src.federated.merge import lora_to_delta
from types import SimpleNamespace


def test_centering_preserves_softmax_and_does_not_increase_rank():
    gen = torch.Generator().manual_seed(123)
    state = {'fc': {'A': torch.randn(4, 9, generator=gen, dtype=torch.float64),
                    'B': torch.randn(7, 4, generator=gen, dtype=torch.float64)}}
    centered = centered_state(state)
    x = torch.randn(31, 9, generator=gen, dtype=torch.float64)
    before, after = lora_to_delta(state, 32)['fc'], lora_to_delta(centered, 32)['fc']
    torch.testing.assert_close((x @ before.T).softmax(1), (x @ after.T).softmax(1), atol=1e-12, rtol=1e-12)
    assert torch.linalg.matrix_rank(after) <= 4
    assert matrix_diagnostics(after)['common_logit_energy_fraction'] < 1e-28
    torch.testing.assert_close(centered['fc']['B'].mean(0), torch.zeros(4, dtype=torch.float64), atol=1e-14, rtol=0)


def test_iid_preserves_ownership_sizes_union_and_test_shards():
    labels = np.tile(np.arange(100), 100)
    config = SimpleNamespace(n_domains=5, clients_per_domain=3, dirichlet_alpha=.5,
                             max_train_per_client=0, max_test_per_domain=0)
    original, assignments = make_splits(labels, labels, config, 42)
    changed, new_assignments = iid_splits(labels, labels, config, 42)
    assert assignments == new_assignments
    assert sorted(i for s in changed.values() for i in s['train_indices']) == list(range(len(labels)))
    for cid in original:
        assert len(original[cid]['train_indices']) == len(changed[cid]['train_indices'])
        assert original[cid]['test_indices'] == changed[cid]['test_indices']
        assert sum(n > 0 for n in changed[cid]['train_class_counts']) > 80
    assert changed == iid_splits(labels, labels, config, 42)[0]
