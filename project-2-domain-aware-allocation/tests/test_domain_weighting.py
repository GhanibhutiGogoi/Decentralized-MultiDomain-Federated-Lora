import numpy as np
import pytest

from framework.aggregation.domain_weighting import conservative_domain_factors
from experiment2.lambda_aggregation import normalized_conservative_domain_weights


def test_no_features_preserves_base_weights():
    samples = [10, 20, 30]
    quality = [0.5, 1.0, 2.0]
    factors = conservative_domain_factors(samples, quality)
    assert np.allclose(factors, 1.0)
    weights = normalized_conservative_domain_weights(samples, quality)
    expected = np.asarray(samples) * np.asarray(quality)
    expected /= expected.sum()
    assert np.allclose(weights, expected)


def test_factors_are_bounded_and_base_weight_normalized():
    samples = [10, 20, 30, 40]
    quality = [1, 1, 1, 1]
    features = {"js": [0.0, 0.1, 1.0, 100.0], "entropy": [1, 2, 3, 4]}
    factors = conservative_domain_factors(samples, quality, features)
    assert np.all(factors >= 0.85 - 1e-8)
    assert np.all(factors <= 1.15 + 1e-8)
    assert np.isclose(np.average(factors, weights=samples), 1.0)
    assert factors[-1] > factors[0]


def test_invalid_lengths_and_values_fail_closed():
    with pytest.raises(ValueError, match="exactly 2"):
        conservative_domain_factors([1, 2], [1], {"x": [1, 2]})
    with pytest.raises(ValueError, match="finite"):
        conservative_domain_factors([1, 2], [1, 1], {"x": [1, np.nan]})
    with pytest.raises(ValueError, match="positive total"):
        conservative_domain_factors([1, 2], [0, 0], {"x": [1, 2]})


def test_constant_feature_is_identity():
    factors = conservative_domain_factors([1, 1, 1], [1, 1, 1], {"x": [4, 4, 4]})
    assert np.allclose(factors, 1.0)
