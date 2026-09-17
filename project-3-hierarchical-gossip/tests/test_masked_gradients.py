import itertools

import torch

from experiments.explore_masked_gradients import partial_gradients, padded_gradient


def test_partial_gradient_matches_active_autograd_coordinates():
    gen = torch.Generator().manual_seed(9)
    x = torch.randn(13, 11, generator=gen, dtype=torch.float64)
    labels = torch.randint(7, (13,), generator=gen)
    a = torch.randn(4, 11, generator=gen, dtype=torch.float64, requires_grad=True)
    b = torch.randn(7, 4, generator=gen, dtype=torch.float64, requires_grad=True)
    base = torch.randn(7, 11, generator=gen, dtype=torch.float64)
    bias = torch.randn(7, generator=gen, dtype=torch.float64)
    loss = torch.nn.functional.cross_entropy(x @ base.T + bias + (x @ a.T @ b.T) * 8, labels)
    loss.backward()
    selected = torch.tensor([3, 1])
    da, db, observed = partial_gradients(x, labels, a, b, base, bias, selected)
    torch.testing.assert_close(da, a.grad[selected], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(db, b.grad[:, selected], atol=1e-12, rtol=1e-12)
    assert abs(observed - float(loss)) < 1e-12


def test_importance_corrected_coordinate_gradient_is_unbiased():
    gen = torch.Generator().manual_seed(13)
    a = torch.randn(4, 5, generator=gen, dtype=torch.float64)
    b = torch.randn(3, 4, generator=gen, dtype=torch.float64)
    full_a, full_b = torch.randn(4, 5, generator=gen, dtype=torch.float64), torch.randn(3, 4, generator=gen, dtype=torch.float64)
    for rank in [1, 2, 3, 4]:
        subsets = list(itertools.combinations(range(4), rank))
        expected = torch.stack([padded_gradient(a, b, list(s), full_a[list(s)], full_b[:, list(s)], 4 / rank) for s in subsets]).mean(0)
        torch.testing.assert_close(expected, torch.cat([full_a.flatten(), full_b.flatten()]), atol=1e-12, rtol=1e-12)
