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
from experiment2.lambda_calibration import FORM_A_FEATURES, FORM_B_FEATURES, LinearFit  # noqa: E402
from experiment3.calibration_bundle import load_calibration_bundle  # noqa: E402


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
    assert loaded.schema_version == "exp3-calibration-bundle/v1"
    assert loaded.purpose == "scientific_calibration"
    assert loaded.dataset_provenance["TinyTask"]["is_synthetic"] is False


def test_experiment2_writer_refuses_synthetic_or_incomplete_source():
    measurements, exp1_manifest, exp1_dataset_manifest = _inputs(is_synthetic=True)
    with pytest.raises(CalibrationBundleWriteError, match="real-data"):
        build_calibration_bundle(
            measurements=measurements,
            exp1_manifest=exp1_manifest,
            exp1_dataset_manifest=exp1_dataset_manifest,
            fits=[_fit("form_a", FORM_A_FEATURES), _fit("form_b", FORM_B_FEATURES)],
            lambda_calibrations={"form_a": {"gamma": 0.2}, "form_b": {"gamma": 0.3}},
            experiment1_run_id="exp1",
            experiment2_run_id="exp2",
        )
