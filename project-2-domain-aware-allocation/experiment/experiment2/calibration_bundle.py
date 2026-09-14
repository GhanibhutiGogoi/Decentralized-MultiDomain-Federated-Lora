"""Write the strict Experiment 3 calibration bundle from Experiment 2 output."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from experiment2.lambda_calibration import LAMBDA_MAX, LAMBDA_MIN, LinearFit
from experiment2.provenance import parse_provenance_value
from experiment3.calibration_bundle import EXPECTED_FORM_FEATURES, SCHEMA_VERSION


class CalibrationBundleWriteError(ValueError):
    """Raised when Experiment 2 cannot produce a trusted calibration bundle."""


def stable_payload_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write JSON via temp file, fsync, and atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _finite_float(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise CalibrationBundleWriteError(f"{label} must be numeric.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationBundleWriteError(f"{label} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise CalibrationBundleWriteError(f"{label} must be finite.")
    if positive and parsed <= 0.0:
        raise CalibrationBundleWriteError(f"{label} must be positive.")
    return parsed


def _source_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _expected_tasks(measurements: pd.DataFrame, exp1_manifest: Mapping[str, object]) -> list[str]:
    manifest_tasks = exp1_manifest.get("tasks")
    if isinstance(manifest_tasks, Sequence) and not isinstance(manifest_tasks, (str, bytes)):
        tasks = [str(task) for task in manifest_tasks]
    elif "task" in measurements.columns:
        tasks = list(dict.fromkeys(str(task) for task in measurements["task"].tolist()))
    else:
        raise CalibrationBundleWriteError("Cannot determine calibrated task set.")
    if not tasks or len(set(tasks)) != len(tasks):
        raise CalibrationBundleWriteError("Calibrated task set must be non-empty and unique.")
    measured = {str(task) for task in measurements["task"].unique()}
    if set(tasks) != measured:
        raise CalibrationBundleWriteError("Experiment 1 manifest tasks do not match measurements.")
    return tasks


def _dataset_record(task: str, exp1_dataset_manifest: Mapping[str, object]) -> dict[str, object]:
    datasets = exp1_dataset_manifest.get("datasets")
    if not isinstance(datasets, Mapping) or task not in datasets:
        raise CalibrationBundleWriteError(f"Dataset manifest is missing task {task!r}.")
    raw = datasets[task]
    if not isinstance(raw, Mapping):
        raise CalibrationBundleWriteError(f"Dataset manifest record for {task!r} is malformed.")
    parsed_synthetic = parse_provenance_value(raw.get("is_synthetic", raw.get("synthetic")))
    if parsed_synthetic is None:
        raise CalibrationBundleWriteError(f"Dataset provenance for {task!r} has ambiguous synthetic status.")
    if parsed_synthetic:
        raise CalibrationBundleWriteError(f"Dataset provenance for {task!r} is synthetic.")
    dataset_id = raw.get("dataset_id") or raw.get("name") or raw.get("display_name") or raw.get("source_library")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        dataset_id = task
    return {
        "is_synthetic": False,
        "dataset_id": dataset_id,
        "source_record": dict(raw),
    }


def _dataset_provenance(
    tasks: Sequence[str],
    exp1_dataset_manifest: Mapping[str, object],
) -> dict[str, object]:
    return {task: _dataset_record(task, exp1_dataset_manifest) for task in tasks}


def _form_payload(
    fit: LinearFit,
    lambda_calibrations: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    expected = EXPECTED_FORM_FEATURES.get(fit.form)
    if tuple(fit.features) != tuple(expected or ()):
        raise CalibrationBundleWriteError(f"{fit.form} feature order does not match Experiment 3 schema.")
    if fit.form not in lambda_calibrations:
        raise CalibrationBundleWriteError(f"Missing lambda calibration for {fit.form}.")
    coefficients = [
        _finite_float(value, f"{fit.form}.coefficient[{idx}]")
        for idx, value in enumerate(fit.coefficients)
    ]
    if len(coefficients) != len(fit.features):
        raise CalibrationBundleWriteError(f"{fit.form} coefficients do not match feature order.")
    return {
        "feature_order": list(fit.features),
        "coefficients": coefficients,
        "intercept": _finite_float(fit.intercept, f"{fit.form}.intercept"),
        "gamma": _finite_float(lambda_calibrations[fit.form].get("gamma"), f"{fit.form}.gamma"),
        "clipping_bounds": [float(LAMBDA_MIN), float(LAMBDA_MAX)],
        "standardization_means": {
            feature: _finite_float(fit.feature_means[feature], f"{fit.form}.mean.{feature}")
            for feature in fit.features
        },
        "standardization_stds": {
            feature: _finite_float(
                fit.feature_stds[feature],
                f"{fit.form}.std.{feature}",
                positive=True,
            )
            for feature in fit.features
        },
    }


def build_calibration_bundle(
    *,
    measurements: pd.DataFrame,
    exp1_manifest: Mapping[str, object],
    exp1_dataset_manifest: Mapping[str, object],
    fits: Sequence[LinearFit],
    lambda_calibrations: Mapping[str, Mapping[str, object]],
    experiment1_run_id: str,
    experiment2_run_id: str,
) -> dict[str, object]:
    """Build the completed production calibration contract for Experiment 3."""
    if "is_synthetic" not in measurements.columns:
        raise CalibrationBundleWriteError("Measurements are missing is_synthetic provenance.")
    synthetic_values = measurements["is_synthetic"].map(parse_provenance_value)
    if synthetic_values.isna().any() or synthetic_values.astype(bool).any():
        raise CalibrationBundleWriteError("Calibration bundle requires complete real-data measurements.")
    tasks = _expected_tasks(measurements, exp1_manifest)
    form_payloads = {
        fit.form: _form_payload(fit, lambda_calibrations)
        for fit in fits
    }
    if set(form_payloads) != set(EXPECTED_FORM_FEATURES):
        raise CalibrationBundleWriteError("Calibration bundle requires exactly Form A and Form B.")
    if not experiment1_run_id or not experiment2_run_id:
        raise CalibrationBundleWriteError("Experiment run identities are required.")
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": "scientific_calibration",
        "scientifically_valid": True,
        "is_synthetic": False,
        "completion_status": "complete",
        "expected_task_set": tasks,
        "source": {
            "experiment1_run_id": str(experiment1_run_id),
            "experiment2_run_id": str(experiment2_run_id),
            "commit_sha": _source_commit(),
            "code_revision": stable_payload_hash(
                {
                    "experiment1_run_id": str(experiment1_run_id),
                    "experiment2_run_id": str(experiment2_run_id),
                    "tasks": tasks,
                    "forms": form_payloads,
                }
            ),
        },
        "dataset_provenance": _dataset_provenance(tasks, exp1_dataset_manifest),
        "forms": form_payloads,
    }


def write_calibration_bundle(path: Path, payload: Mapping[str, object]) -> None:
    atomic_write_json(path, payload)
