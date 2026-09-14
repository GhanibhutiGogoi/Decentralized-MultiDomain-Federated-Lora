import json
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
    NULL_IMPROVEMENT_TOLERANCE,
    _support_decision,
    permutation_rank_p_value,
)
from experiment2.form_c import (  # noqa: E402
    FORM_C_FEATURES,
    fit_form_c,
    predict_form_c_score,
    transform_form_c_context,
)
from experiment2.lambda_calibration import lambda_weights_from_scores  # noqa: E402
from experiment3.arms import (  # noqa: E402
    ClientRecord,
    Experiment3ArmError,
    calculate_lambda_weights,
)
from experiment3.calibration_bundle import load_calibration_bundle  # noqa: E402


def _form_c_frame() -> pd.DataFrame:
    rows = []
    for task in ["T1", "T2"]:
        for round_id in [1, 2]:
            for idx, value in enumerate([1.0, 2.0, 3.0]):
                rows.append(
                    {
                        "task": task,
                        "round": round_id,
                        "client_id": f"c{idx}",
                        "delta_accuracy": value,
                        "log_update_l2": value,
                        "js_to_global": 10.0,
                        "update_cosine_distance_to_mean": 4.0 - value,
                        "normalized_entropy": 0.5 + idx,
                        "log_class_imbalance_ratio": 1.5,
                    }
                )
    return pd.DataFrame(rows)


def test_form_c_transform_centers_scales_and_preserves_rows():
    df = _form_c_frame()
    out = transform_form_c_context(df)

    assert list(out.frame["client_id"]) == list(df["client_id"])
    assert len(out.frame) == len(df)
    for _, group in out.frame.groupby(["task", "round"]):
        assert abs(group["form_c__log_update_l2"].mean()) < 1e-12
        assert abs(group["form_c__log_update_l2"].std(ddof=0) - 1.0) < 1e-12
        assert abs(group["form_c__target"].mean()) < 1e-12
        assert abs(group["form_c__target"].std(ddof=0) - 1.0) < 1e-12
        assert np.allclose(group["form_c__js_to_global"], 0.0)
        assert np.allclose(group["form_c__log_class_imbalance_ratio"], 0.0)
    assert out.feature_zero_variance_counts["js_to_global"] == 4
    assert out.feature_zero_variance_counts["log_class_imbalance_ratio"] == 4
    assert out.target_zero_variance_group_count == 0


def test_form_c_zero_variance_target_group_is_retained_as_zero():
    df = _form_c_frame()
    df.loc[df["task"] == "T1", "delta_accuracy"] = 2.0
    out = transform_form_c_context(df)

    assert len(out.frame) == len(df)
    assert out.target_zero_variance_group_count == 2
    assert np.allclose(out.frame.loc[out.frame["task"] == "T1", "form_c__target"], 0.0)


def test_form_c_row_ordering_does_not_change_scores_after_realignment():
    df = _form_c_frame()
    fit = fit_form_c(df, 1.0)
    original = pd.Series(predict_form_c_score(df, fit), index=df["client_id"] + df["task"] + df["round"].astype(str))
    shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    shuffled_scores = pd.Series(
        predict_form_c_score(shuffled, fit),
        index=shuffled["client_id"] + shuffled["task"] + shuffled["round"].astype(str),
    )

    assert np.allclose(original.sort_index(), shuffled_scores.sort_index())


def test_held_out_targets_do_not_affect_form_c_prediction():
    train = _form_c_frame().query("task == 'T1'").reset_index(drop=True)
    test = _form_c_frame().query("task == 'T2'").reset_index(drop=True)
    fit = fit_form_c(train, 1.0)
    before = predict_form_c_score(test, fit)
    changed = test.copy()
    changed["delta_accuracy"] = [100.0, -100.0, 50.0, -50.0, 25.0, -25.0]
    after = predict_form_c_score(changed, fit)

    assert np.allclose(before, after)


def test_form_c_support_gate_accepts_interior_and_rejects_bad_cases():
    accepted = _support_decision(
        form="form_c",
        model_mean_rmse=0.8,
        null_mean_rmse=1.0,
        selected_alpha=1.0,
        alpha_boundary_status="interior",
    )
    boundary = _support_decision(
        form="form_c",
        model_mean_rmse=0.8,
        null_mean_rmse=1.0,
        selected_alpha=100.0,
        alpha_boundary_status="maximum",
    )
    worse = _support_decision(
        form="form_c",
        model_mean_rmse=1.1,
        null_mean_rmse=1.0,
        selected_alpha=1.0,
        alpha_boundary_status="interior",
    )
    equivalent = _support_decision(
        form="form_c",
        model_mean_rmse=1.0 - NULL_IMPROVEMENT_TOLERANCE / 2.0,
        null_mean_rmse=1.0,
        selected_alpha=1.0,
        alpha_boundary_status="interior",
    )
    reversed_orientation = _support_decision(
        form="form_c",
        model_mean_rmse=0.8,
        null_mean_rmse=1.0,
        selected_alpha=1.0,
        alpha_boundary_status="interior",
        reversed_orientation=True,
    )

    assert accepted.supported
    assert not boundary.supported
    assert boundary.rejection_reason == "ridge_alpha_boundary_maximum"
    assert not worse.supported
    assert not equivalent.supported
    assert not reversed_orientation.supported


def test_form_c_lambda_uses_group_centered_scores():
    df = pd.DataFrame({"task": ["T"] * 3, "round": [1] * 3})
    weights = lambda_weights_from_scores(df, [-1.0, 0.0, 1.0], 0.2)

    assert len(weights) == 3
    assert abs(float(np.mean(weights)) - 1.0) < 1e-12
    assert all(0.5 <= float(value) <= 1.5 for value in weights)


def test_stratified_permutation_rejects_group_length_mismatch_and_preserves_groups():
    y = [0.0, 1.0, 0.0, 1.0]
    score = [0.0, 1.0, 0.0, 1.0]
    p_value = permutation_rank_p_value(
        y,
        score,
        groups=[("T", 1), ("T", 1), ("T", 2), ("T", 2)],
        n_permutations=5,
        seed=1,
    )
    assert 0.0 < p_value <= 1.0
    with pytest.raises(ValueError):
        permutation_rank_p_value(y, score, groups=[("T", 1)], n_permutations=5)


def test_supported_form_c_bundle_loads_and_scores_without_target_feature(tmp_path):
    bundle = {
        "schema_version": "exp3-calibration-bundle/v3",
        "purpose": "engineering_test_only",
        "scientifically_valid": False,
        "is_synthetic": True,
        "completion_status": "engineering_fixture",
        "expected_task_set": ["TinyTask"],
        "baseline_arm": "baseline",
        "supported_treatment_arms": ["form_c"],
        "source": {
            "experiment1_run_id": "exp1",
            "experiment2_run_id": "exp2",
            "commit_sha": "0" * 40,
        },
        "dataset_provenance": {
            "TinyTask": {"dataset_id": "tiny", "is_synthetic": False}
        },
        "forms": {
            "form_c": {
                "feature_order": FORM_C_FEATURES,
                "coefficients": [0.2, 0.1, -0.1, 0.05, 0.03],
                "intercept": 0.0,
                "ridge_alpha": 1.0,
                "gamma": 0.2,
                "clipping_bounds": [0.5, 1.5],
                "standardization_means": {feature: 0.0 for feature in FORM_C_FEATURES},
                "standardization_stds": {feature: 1.0 for feature in FORM_C_FEATURES},
                "methodology_label": "Form C - within-round relative-contribution calibration",
                "feature_transformation": "within_task_round_population_zscore",
                "target_transformation": "within_task_round_zscore_delta_accuracy",
                "exploratory_status": "post_hoc_exploratory",
            }
        },
        "unsupported_forms": [],
    }
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    loaded = load_calibration_bundle(
        path,
        expected_tasks=["TinyTask"],
        expected_experiment1_run_id="exp1",
        expected_experiment2_run_id="exp2",
        allow_engineering_fixture=True,
    )
    records = [
        ClientRecord(
            "TinyTask",
            1,
            "c0",
            10,
            1.0,
            dict(zip(FORM_C_FEATURES, [1.0, 0.1, 0.3, 0.7, 0.2])),
        ),
        ClientRecord(
            "TinyTask",
            1,
            "c1",
            10,
            1.0,
            dict(zip(FORM_C_FEATURES, [2.0, 0.2, 0.2, 0.8, 0.3])),
        ),
        ClientRecord(
            "TinyTask",
            1,
            "c2",
            10,
            1.0,
            dict(zip(FORM_C_FEATURES, [3.0, 0.3, 0.1, 0.9, 0.4])),
        ),
    ]
    weights = calculate_lambda_weights(records, loaded, "form_c")

    assert len(weights) == 3
    assert abs(float(np.mean(weights)) - 1.0) < 1e-12
    with pytest.raises(Experiment3ArmError):
        calculate_lambda_weights(records, loaded, "form_b")
