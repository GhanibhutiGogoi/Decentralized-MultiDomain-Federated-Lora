import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT2_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT2_ROOT, PROJECT2_ROOT / "experiment"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment2.calibration_bundle import (  # noqa: E402
    CalibrationBundleWriteError,
    build_calibration_bundle,
    write_calibration_bundle,
)
from experiment2.form_c import FORM_C_FEATURES, fit_form_c  # noqa: E402
from experiment2.lambda_calibration import FORM_A_FEATURES, FORM_B_FEATURES, LinearFit  # noqa: E402
from experiment3.calibration_bundle import SCHEMA_VERSION_V3, load_calibration_bundle  # noqa: E402


def _fit(form, features):
    return LinearFit(
        form=form,
        features=list(features),
        coefficients=np.arange(1, len(features) + 1, dtype=float) / 10.0,
        intercept=0.1,
        feature_means={feature: 0.0 for feature in features},
        feature_stds={feature: 1.0 for feature in features},
        target_mean=0.0,
        target_std=1.0,
    )


def _inputs(is_synthetic=False):
    measurements = pd.DataFrame(
        {
            "task": ["TinyTask", "TinyTask"],
            "is_synthetic": [is_synthetic, is_synthetic],
        }
    )
    exp1_manifest = {"tasks": ["TinyTask"]}
    exp1_dataset_manifest = {
        "datasets": {
            "TinyTask": {
                "synthetic": is_synthetic,
                "dataset_id": "tiny-real-dataset",
            }
        }
    }
    return measurements, exp1_manifest, exp1_dataset_manifest


def test_experiment2_writer_produces_exp3_loadable_bundle(tmp_path):
    measurements, exp1_manifest, exp1_dataset_manifest = _inputs()
    payload = build_calibration_bundle(
        measurements=measurements,
        exp1_manifest=exp1_manifest,
        exp1_dataset_manifest=exp1_dataset_manifest,
        fits=[_fit("form_a", FORM_A_FEATURES), _fit("form_b", FORM_B_FEATURES)],
        lambda_calibrations={"form_a": {"gamma": 0.2}, "form_b": {"gamma": 0.3}},
        experiment1_run_id="exp1-real",
        experiment2_run_id="exp2-real",
    )
    path = tmp_path / "calibration_bundle.json"
    write_calibration_bundle(path, payload)
    loaded = load_calibration_bundle(
        path,
        expected_tasks=["TinyTask"],
        expected_experiment1_run_id="exp1-real",
        expected_experiment2_run_id="exp2-real",
    )
    assert loaded.schema_version == SCHEMA_VERSION_V3
    assert loaded.purpose == "scientific_calibration"
    assert loaded.dataset_provenance["TinyTask"]["is_synthetic"] is False
    assert loaded.supported_treatment_arms == ("form_a", "form_b")


def test_experiment2_writer_can_emit_only_supported_form_without_other_parameters(tmp_path):
    measurements, exp1_manifest, exp1_dataset_manifest = _inputs()
    form_support = pd.DataFrame(
        [
            {
                "form": "form_a",
                "status": "supported",
                "model_mean_rmse": 0.1,
                "null_mean_rmse": 0.2,
                "rejection_reason": "",
                "selected_alpha": "",
                "alpha_boundary_status": "not_applicable",
            },
            {
                "form": "form_b",
                "status": "unsupported",
                "model_mean_rmse": 0.21,
                "null_mean_rmse": 0.2,
                "rejection_reason": "ridge_alpha_boundary_maximum",
                "selected_alpha": 100000.0,
                "alpha_boundary_status": "maximum",
            },
        ]
    )
    payload = build_calibration_bundle(
        measurements=measurements,
        exp1_manifest=exp1_manifest,
        exp1_dataset_manifest=exp1_dataset_manifest,
        fits=[_fit("form_a", FORM_A_FEATURES)],
        lambda_calibrations={"form_a": {"gamma": 0.2}},
        experiment1_run_id="exp1-real",
        experiment2_run_id="exp2-real",
        form_support=form_support,
    )
    assert payload["schema_version"] == SCHEMA_VERSION_V3
    assert payload["supported_treatment_arms"] == ["form_a"]
    assert set(payload["forms"]) == {"form_a"}
    assert "coefficients" not in payload["unsupported_forms"][0]

    path = tmp_path / "calibration_bundle.json"
    write_calibration_bundle(path, payload)
    loaded = load_calibration_bundle(
        path,
        expected_tasks=["TinyTask"],
        expected_experiment1_run_id="exp1-real",
        expected_experiment2_run_id="exp2-real",
    )
    assert loaded.supported_treatment_arms == ("form_a",)
    assert set(loaded.forms) == {"form_a"}


def test_experiment2_writer_round_trips_supported_form_c(tmp_path):
    measurements = pd.DataFrame(
        {
            "task": ["TinyTask"] * 6,
            "round": [1, 1, 1, 2, 2, 2],
            "client_id": ["a", "b", "c", "a", "b", "c"],
            "is_synthetic": [False] * 6,
            "delta_accuracy": [0.1, 0.2, 0.3, 0.3, 0.1, 0.2],
            "log_update_l2": [1.0, 2.0, 3.0, 3.0, 1.0, 2.0],
            "js_to_global": [0.2, 0.3, 0.4, 0.4, 0.2, 0.3],
            "update_cosine_distance_to_mean": [0.1, 0.2, 0.3, 0.3, 0.1, 0.2],
            "normalized_entropy": [0.8, 0.7, 0.6, 0.6, 0.8, 0.7],
            "log_class_imbalance_ratio": [0.0, 0.1, 0.2, 0.2, 0.0, 0.1],
        }
    )
    exp1_manifest = {"tasks": ["TinyTask"]}
    exp1_dataset_manifest = {
        "datasets": {"TinyTask": {"synthetic": False, "dataset_id": "tiny-real-dataset"}}
    }
    fit = fit_form_c(measurements, 1.0)
    payload = build_calibration_bundle(
        measurements=measurements,
        exp1_manifest=exp1_manifest,
        exp1_dataset_manifest=exp1_dataset_manifest,
        fits=[fit],
        lambda_calibrations={"form_c": {"gamma": 0.2}},
        experiment1_run_id="exp1-real",
        experiment2_run_id="exp2-real",
    )
    assert payload["schema_version"] == SCHEMA_VERSION_V3
    assert payload["supported_treatment_arms"] == ["form_c"]
    assert payload["forms"]["form_c"]["feature_order"] == FORM_C_FEATURES
    assert payload["forms"]["form_c"]["feature_transformation"] == "within_task_round_population_zscore"

    path = tmp_path / "calibration_bundle.json"
    write_calibration_bundle(path, payload)
    loaded = load_calibration_bundle(
        path,
        expected_tasks=["TinyTask"],
        expected_experiment1_run_id="exp1-real",
        expected_experiment2_run_id="exp2-real",
    )
    assert loaded.supported_treatment_arms == ("form_c",)
    assert loaded.forms["form_c"].target_transformation == "within_task_round_zscore_delta_accuracy"


def test_experiment2_writer_refuses_synthetic_or_incomplete_source():
    measurements, exp1_manifest, exp1_dataset_manifest = _inputs(is_synthetic=True)
    with pytest.raises(CalibrationBundleWriteError, match="real-data"):
        build_calibration_bundle(
            measurements=measurements,
            exp1_manifest=exp1_manifest,
            exp1_dataset_manifest=exp1_dataset_manifest,
            fits=[_fit("form_a", FORM_A_FEATURES)],
            lambda_calibrations={"form_a": {"gamma": 0.2}},
            experiment1_run_id="exp1",
            experiment2_run_id="exp2",
        )
