import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT2_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT2_ROOT, PROJECT2_ROOT / "experiment"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment2.evaluation import (  # noqa: E402
    EvaluationConfig,
    support_decision_evaluation,
)
from experiment2.lambda_calibration import ridge_alpha_grid  # noqa: E402


def _prepared_measurements() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(3)
    for task_idx, task in enumerate(["A", "B", "C", "D"]):
        offset = [1.0, -1.0, 0.5, -0.5][task_idx]
        for client_idx in range(5):
            rows.append(
                {
                    "task": task,
                    "round": 1,
                    "client_id": f"c{client_idx}",
                    "delta_accuracy": offset + rng.normal(0.0, 0.01),
                    "log_update_l2": rng.normal(),
                    "js_to_global": rng.normal(),
                    "update_cosine_distance_to_mean": rng.normal(),
                    "normalized_entropy": rng.normal(),
                    "log_class_imbalance_ratio": rng.normal(),
                }
            )
    return pd.DataFrame(rows)


def test_support_decision_records_fold_safe_null_and_boundary_without_raising():
    cv, form_support, decisions = support_decision_evaluation(
        _prepared_measurements(),
        ridge_alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        config=EvaluationConfig(permutations=0),
        source_run_identity="exp1-test",
    )

    assert set(cv["form"]) == {"null", "form_a", "form_b", "form_c"}
    assert list(form_support["form"]) == ["form_a", "form_b", "form_c"]
    assert decisions["form_a"].supported
    assert not decisions["form_b"].supported
    assert decisions["form_b"].selected_alpha == 1000.0
    assert decisions["form_b"].alpha_boundary_status == "maximum"
    assert not bool(
        form_support.query("form == 'form_b'")["coefficient_available"].iloc[0]
    )
    assert not bool(
        form_support.query("form == 'form_b'")["gamma_lambda_available"].iloc[0]
    )


@pytest.mark.parametrize("bad_alpha", [math.nan, math.inf, -math.inf, True, False])
def test_custom_ridge_alpha_grid_rejects_non_finite_and_bool_values(bad_alpha):
    with pytest.raises(ValueError):
        ridge_alpha_grid(custom_alphas=[0.1, bad_alpha, 1.0])


def test_custom_ridge_alpha_grid_sorts_and_deduplicates_positive_values():
    assert ridge_alpha_grid(custom_alphas=[10, 0.1, 10.0, 1]) == [0.1, 1.0, 10.0]
