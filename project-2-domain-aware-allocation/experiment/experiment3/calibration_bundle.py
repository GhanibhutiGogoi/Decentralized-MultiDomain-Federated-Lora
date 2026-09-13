"""Strict calibration-bundle contract for Experiment 3.

Experiment 3 consumes a future Experiment 2 calibration bundle at runtime. This
loader intentionally fails closed: engineering fixtures are accepted only by an
explicit private test hook, never by normal production execution.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA_VERSION = "exp3-calibration-bundle/v1"
FORMS = ("form_a", "form_b")
EXPECTED_FORM_FEATURES = {
    "form_a": ("log_update_l2", "js_to_global"),
    "form_b": (
        "log_update_l2",
        "js_to_global",
        "update_cosine_distance_to_mean",
        "normalized_entropy",
        "log_class_imbalance_ratio",
    ),
}


class CalibrationBundleError(ValueError):
    """Raised when a calibration bundle cannot be trusted."""


@dataclass(frozen=True)
class FormCalibration:
    """Validated calibration for one lambda form."""

    form: str
    feature_order: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    gamma: float
    clipping_bounds: tuple[float, float]
    standardization_means: Mapping[str, float]
    standardization_stds: Mapping[str, float]


@dataclass(frozen=True)
class CalibrationBundle:
    """Validated Experiment 2 calibration contract consumed by Experiment 3."""

    path: Path
    content_hash: str
    schema_version: str
    purpose: str
    scientifically_valid: bool
    is_synthetic: bool
    completion_status: str
    expected_task_set: tuple[str, ...]
    source_experiment1_run_id: str
    source_experiment2_run_id: str
    source_commit_sha: str
    dataset_provenance: Mapping[str, object]
    forms: Mapping[str, FormCalibration]


def _load_json_without_duplicate_keys(path: Path) -> dict:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict:
        seen = set()
        out = {}
        for key, value in pairs:
            if key in seen:
                raise CalibrationBundleError(
                    f"{path} contains duplicate JSON key {key!r}."
                )
            seen.add(key)
            out[key] = value
        return out

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except json.JSONDecodeError as exc:
        raise CalibrationBundleError(f"{path} is not valid JSON: {exc}") from exc


def _required_mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise CalibrationBundleError(f"Calibration bundle is missing object {key!r}.")
    return value


def _required_string(parent: Mapping[str, object], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CalibrationBundleError(f"Calibration bundle is missing string {key!r}.")
    return value


def _required_bool(parent: Mapping[str, object], key: str) -> bool:
    value = parent.get(key)
    if not isinstance(value, bool):
        raise CalibrationBundleError(
            f"Calibration bundle field {key!r} must be a strict boolean."
        )
    return value


def _finite_float(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or value is None:
        raise CalibrationBundleError(f"{label} must be a finite numeric value.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationBundleError(f"{label} must be a finite numeric value.") from exc
    if not math.isfinite(parsed):
        raise CalibrationBundleError(f"{label} must be finite.")
    if positive and parsed <= 0.0:
        raise CalibrationBundleError(f"{label} must be positive.")
    return parsed


def _required_task_set(data: Mapping[str, object]) -> tuple[str, ...]:
    raw = data.get("expected_task_set")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise CalibrationBundleError("expected_task_set must be a non-empty list.")
    tasks = []
    seen = set()
    for value in raw:
        if not isinstance(value, str) or not value.strip():
            raise CalibrationBundleError("expected_task_set contains an invalid task.")
        if value in seen:
            raise CalibrationBundleError(
                f"expected_task_set contains duplicate task {value!r}."
            )
        seen.add(value)
        tasks.append(value)
    return tuple(tasks)


def _validate_dataset_provenance(
    provenance: Mapping[str, object],
    expected_tasks: tuple[str, ...],
) -> None:
    if set(provenance) != set(expected_tasks):
        raise CalibrationBundleError(
            "dataset_provenance task set does not match expected_task_set."
        )
    for task in expected_tasks:
        record = provenance[task]
        if not isinstance(record, Mapping):
            raise CalibrationBundleError(
                f"dataset_provenance for task {task!r} must be an object."
            )
        if "is_synthetic" not in record:
            raise CalibrationBundleError(
                f"dataset_provenance for task {task!r} is missing is_synthetic."
            )
        if not isinstance(record["is_synthetic"], bool):
            raise CalibrationBundleError(
                f"dataset_provenance for task {task!r} uses ambiguous is_synthetic."
            )
        if record["is_synthetic"]:
            raise CalibrationBundleError(
                f"dataset_provenance identifies synthetic task {task!r}."
            )
        if not isinstance(record.get("dataset_id"), str) or not record["dataset_id"]:
            raise CalibrationBundleError(
                f"dataset_provenance for task {task!r} is missing dataset_id."
            )


def _parse_form(form: str, raw: object) -> FormCalibration:
    if form not in EXPECTED_FORM_FEATURES:
        raise CalibrationBundleError(f"Unsupported calibration form {form!r}.")
    if not isinstance(raw, Mapping):
        raise CalibrationBundleError(f"{form} calibration must be an object.")

    feature_order_raw = raw.get("feature_order")
    expected_features = EXPECTED_FORM_FEATURES[form]
    if tuple(feature_order_raw or ()) != expected_features:
        raise CalibrationBundleError(
            f"{form} feature_order must exactly equal {list(expected_features)!r}."
        )

    coefficients_raw = raw.get("coefficients")
    if (
        not isinstance(coefficients_raw, Sequence)
        or isinstance(coefficients_raw, (str, bytes))
        or len(coefficients_raw) != len(expected_features)
    ):
        raise CalibrationBundleError(
            f"{form} coefficients must align one-to-one with feature_order."
        )
    coefficients = tuple(
        _finite_float(value, f"{form}.coefficients[{idx}]")
        for idx, value in enumerate(coefficients_raw)
    )

    clipping_raw = raw.get("clipping_bounds")
    if (
        not isinstance(clipping_raw, Sequence)
        or isinstance(clipping_raw, (str, bytes))
        or len(clipping_raw) != 2
    ):
        raise CalibrationBundleError(f"{form}.clipping_bounds must have two values.")
    lower = _finite_float(clipping_raw[0], f"{form}.clipping_bounds[0]", positive=True)
    upper = _finite_float(clipping_raw[1], f"{form}.clipping_bounds[1]", positive=True)
    if lower > 1.0 or upper < 1.0 or lower > upper:
        raise CalibrationBundleError(
            f"{form}.clipping_bounds must contain 1.0 and be ordered."
        )

    means_raw = _required_mapping(raw, "standardization_means")
    stds_raw = _required_mapping(raw, "standardization_stds")
    if set(means_raw) != set(expected_features) or set(stds_raw) != set(expected_features):
        raise CalibrationBundleError(
            f"{form} standardization keys must match feature_order exactly."
        )

    means = {
        feature: _finite_float(means_raw[feature], f"{form}.mean.{feature}")
        for feature in expected_features
    }
    stds = {
        feature: _finite_float(
            stds_raw[feature],
            f"{form}.std.{feature}",
            positive=True,
        )
        for feature in expected_features
    }

    return FormCalibration(
        form=form,
        feature_order=expected_features,
        coefficients=coefficients,
        intercept=_finite_float(raw.get("intercept"), f"{form}.intercept"),
        gamma=_finite_float(raw.get("gamma"), f"{form}.gamma"),
        clipping_bounds=(lower, upper),
        standardization_means=means,
        standardization_stds=stds,
    )


def load_calibration_bundle(
    path: Path,
    *,
    expected_tasks: Sequence[str],
    expected_experiment1_run_id: str,
    expected_experiment2_run_id: str,
    allow_engineering_fixture: bool = False,
) -> CalibrationBundle:
    """Load and validate the complete Experiment 2 calibration bundle.

    Production callers must use the default ``allow_engineering_fixture=False``.
    With that default, any synthetic, engineering-only, scientifically invalid,
    incomplete, stale, or run-identity-mismatched bundle is rejected.
    """
    path = Path(path)
    if not path.exists():
        raise CalibrationBundleError(f"Calibration bundle is missing: {path}")
    raw_bytes = path.read_bytes()
    data = _load_json_without_duplicate_keys(path)

    schema_version = _required_string(data, "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise CalibrationBundleError(f"Unsupported schema_version {schema_version!r}.")

    purpose = _required_string(data, "purpose")
    scientifically_valid = _required_bool(data, "scientifically_valid")
    is_synthetic = _required_bool(data, "is_synthetic")
    completion_status = _required_string(data, "completion_status")
    bundle_tasks = _required_task_set(data)
    configured_tasks = tuple(expected_tasks)
    if not configured_tasks or set(configured_tasks) != set(bundle_tasks):
        raise CalibrationBundleError(
            "Calibration bundle task set does not match the Experiment 3 configuration."
        )

    source = _required_mapping(data, "source")
    exp1_id = _required_string(source, "experiment1_run_id")
    exp2_id = _required_string(source, "experiment2_run_id")
    if exp1_id != expected_experiment1_run_id:
        raise CalibrationBundleError("Experiment 1 run identity mismatch.")
    if exp2_id != expected_experiment2_run_id:
        raise CalibrationBundleError("Experiment 2 run identity mismatch.")
    commit_sha = _required_string(source, "commit_sha")
    if "code_revision" in source and not isinstance(source["code_revision"], str):
        raise CalibrationBundleError("source.code_revision must be a string if present.")

    dataset_provenance = _required_mapping(data, "dataset_provenance")
    _validate_dataset_provenance(dataset_provenance, bundle_tasks)

    forms_raw = _required_mapping(data, "forms")
    if set(forms_raw) != set(FORMS):
        raise CalibrationBundleError("Calibration bundle must contain Form A and Form B.")
    forms = {form: _parse_form(form, forms_raw[form]) for form in FORMS}

    is_engineering_fixture = (
        purpose == "engineering_test_only"
        or is_synthetic
        or not scientifically_valid
    )
    if is_engineering_fixture and not allow_engineering_fixture:
        raise CalibrationBundleError(
            "Production Experiment 3 refuses synthetic or engineering-test calibration."
        )
    if not allow_engineering_fixture:
        if purpose != "scientific_calibration":
            raise CalibrationBundleError("Production calibration purpose is unsupported.")
        if not scientifically_valid:
            raise CalibrationBundleError("Production calibration is not scientifically valid.")
        if is_synthetic:
            raise CalibrationBundleError("Production calibration is synthetic.")
        if completion_status != "complete":
            raise CalibrationBundleError("Production calibration is incomplete.")
    elif completion_status not in {"complete", "engineering_fixture"}:
        raise CalibrationBundleError("Engineering calibration completion status is invalid.")

    return CalibrationBundle(
        path=path,
        content_hash=hashlib.sha256(raw_bytes).hexdigest(),
        schema_version=schema_version,
        purpose=purpose,
        scientifically_valid=scientifically_valid,
        is_synthetic=is_synthetic,
        completion_status=completion_status,
        expected_task_set=bundle_tasks,
        source_experiment1_run_id=exp1_id,
        source_experiment2_run_id=exp2_id,
        source_commit_sha=commit_sha,
        dataset_provenance=dataset_provenance,
        forms=forms,
    )

