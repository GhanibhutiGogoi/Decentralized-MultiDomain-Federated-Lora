import math
import sys
from pathlib import Path

import pytest

PROJECT2_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT2_ROOT, PROJECT2_ROOT / "experiment"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment2.lambda_aggregation import (  # noqa: E402
    effective_quality_scores,
    normalized_aggregation_weights,
)


def test_lambda_disabled_preserves_quality_scores():
    assert effective_quality_scores([1, 2, 3], None) == [1.0, 2.0, 3.0]
    assert normalized_aggregation_weights([1, 1, 1], [1, 2, 3]) == [
        1 / 6,
        2 / 6,
        3 / 6,
    ]


@pytest.mark.parametrize(
    "samples,qualities,lambdas,match",
    [
        ([], [], None, "at least one"),
        ([1, 2], [1], None, "equal length"),
        ([1, -1], [1, 1], None, "non-negative"),
        ([1, 1], [1, math.nan], None, "finite"),
        ([1, 1], [1, 1], [1], "equal length"),
        ([0, 0], [1, 1], None, "positive total"),
        ([1, 1], [1, 1], [0, 0], "positive total"),
    ],
)
def test_invalid_weight_inputs_fail_loudly(samples, qualities, lambdas, match):
    with pytest.raises(ValueError, match=match):
        normalized_aggregation_weights(samples, qualities, lambdas)


def test_duplicate_client_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        normalized_aggregation_weights(
            [1, 1],
            [1, 1],
            client_ids=["client_0", "client_0"],
        )
