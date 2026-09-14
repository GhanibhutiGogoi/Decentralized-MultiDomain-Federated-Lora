"""Experiment 3 aggregation support."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from experiment2.form_c import transform_form_c_context
from experiment2.lambda_calibration import lambda_weights_from_scores
from experiment2.lambda_aggregation import normalized_aggregation_weights
from experiment3.calibration_bundle import CalibrationBundle, FormCalibration


ARMS = ("baseline", "form_a", "form_b", "form_c")


class Experiment3ArmError(ValueError):
    """Raised when arm inputs or lambda weights are invalid."""


@dataclass(frozen=True)
class ClientRecord:
    """Features for one client in one task/round."""

    task: str
    round: int
    client_id: str
    sample_count: int
    quality_score: float
    features: Mapping[str, float]


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise Experiment3ArmError(f"{label} must be numeric.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise Experiment3ArmError(f"{label} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise Experiment3ArmError(f"{label} must be finite.")
    if positive and parsed <= 0.0:
        raise Experiment3ArmError(f"{label} must be positive.")
    return parsed


def _score(record: ClientRecord, calibration: FormCalibration) -> float:
    total = calibration.intercept
    for feature, coefficient in zip(
        calibration.feature_order,
        calibration.coefficients,
    ):
        if feature not in record.features:
            raise Experiment3ArmError(
                f"Client {record.client_id!r} is missing feature {feature!r}."
            )
        value = _finite(record.features[feature], f"{record.client_id}.{feature}")
        mean = calibration.standardization_means[feature]
        std = calibration.standardization_stds[feature]
        total += coefficient * ((value - mean) / std)
    return float(total)


def calculate_lambda_weights(
    records: Sequence[ClientRecord],
    bundle: CalibrationBundle,
    form: str,
) -> list[float]:
    """Calculate supported lambda weights preserving client order.

    Scores are standardized in the exact bundle feature order, centered within
    each ``(task, round)`` group, exponentiated with the form-specific gamma,
    clipped to the bundle bounds, and renormalized so each group mean is one.
    """
    if form not in {"form_a", "form_b", "form_c"}:
        raise Experiment3ArmError(f"Unsupported lambda form {form!r}.")
    if form not in bundle.supported_treatment_arms:
        raise Experiment3ArmError(f"Calibration bundle does not support {form!r}.")
    if not records:
        raise Experiment3ArmError("At least one client record is required.")

    calibration = bundle.forms[form]
    groups: dict[tuple[str, int], list[int]] = {}
    seen_ids: set[tuple[str, int, str]] = set()
    for idx, record in enumerate(records):
        key = (record.task, int(record.round), str(record.client_id))
        if key in seen_ids:
            raise Experiment3ArmError(f"Duplicate client identifier {key!r}.")
        seen_ids.add(key)
        if record.task not in bundle.expected_task_set:
            raise Experiment3ArmError(f"Unknown task {record.task!r}.")
        groups.setdefault((record.task, int(record.round)), []).append(idx)

    if form == "form_c":
        feature_rows = []
        for record in records:
            row = {"task": record.task, "round": int(record.round)}
            for feature in calibration.feature_order:
                if feature not in record.features:
                    raise Experiment3ArmError(
                        f"Client {record.client_id!r} is missing feature {feature!r}."
                    )
                row[feature] = _finite(
                    record.features[feature],
                    f"{record.client_id}.{feature}",
                )
            feature_rows.append(row)
        try:
            transformed = transform_form_c_context(
                pd.DataFrame(feature_rows),
                features=calibration.feature_order,
                include_target=False,
            )
            x = transformed.frame[list(transformed.feature_columns)].to_numpy(dtype=float)
            scores = (
                float(calibration.intercept)
                + x @ np.asarray(calibration.coefficients, dtype=float)
            )
        except Exception as exc:
            raise Experiment3ArmError(str(exc)) from exc
        if not np.all(np.isfinite(scores)):
            raise Experiment3ArmError("Form C scores must be finite.")
    else:
        scores = [_score(record, calibration) for record in records]

    group_df = pd.DataFrame(
        {
            "task": [record.task for record in records],
            "round": [int(record.round) for record in records],
        }
    )
    try:
        out = lambda_weights_from_scores(
            group_df,
            scores,
            calibration.gamma,
            lower=calibration.clipping_bounds[0],
            upper=calibration.clipping_bounds[1],
        )
    except Exception as exc:
        raise Experiment3ArmError(str(exc)) from exc
    validate_lambda_weights(records, out)
    return [float(value) for value in out]


def validate_lambda_weights(
    records: Sequence[ClientRecord],
    lambda_weights: Sequence[float],
) -> None:
    """Require one finite non-negative lambda per ordered client record."""
    if len(records) != len(lambda_weights):
        raise Experiment3ArmError("Client records and lambda weights differ in length.")
    seen: set[tuple[str, int, str]] = set()
    for record, value in zip(records, lambda_weights):
        key = (record.task, int(record.round), str(record.client_id))
        if key in seen:
            raise Experiment3ArmError(f"Duplicate client identifier {key!r}.")
        seen.add(key)
        parsed = _finite(value, f"lambda[{key!r}]")
        if parsed < 0.0:
            raise Experiment3ArmError(f"lambda[{key!r}] must be non-negative.")


def realized_aggregation_weights(
    records: Sequence[ClientRecord],
    arm: str,
    bundle: CalibrationBundle | None = None,
) -> list[float]:
    """Return normalized ``w * q`` or ``w * q * lambda`` aggregation weights."""
    if arm not in ARMS:
        raise Experiment3ArmError(f"Unknown Experiment 3 arm {arm!r}.")
    samples = [record.sample_count for record in records]
    qualities = [record.quality_score for record in records]
    for idx, (sample, quality) in enumerate(zip(samples, qualities)):
        _finite(sample, f"sample_count[{idx}]", positive=True)
        _finite(quality, f"quality_score[{idx}]", positive=True)
    if arm == "baseline":
        return normalized_aggregation_weights(
            samples,
            qualities,
            lambda_weights=None,
            client_ids=[record.client_id for record in records],
        )
    if bundle is None:
        raise Experiment3ArmError("Form arms require a calibration bundle.")
    if arm not in bundle.supported_treatment_arms:
        raise Experiment3ArmError(f"Calibration bundle does not support {arm!r}.")
    lambda_weights = calculate_lambda_weights(records, bundle, arm)
    return normalized_aggregation_weights(
        samples,
        qualities,
        lambda_weights,
        client_ids=[record.client_id for record in records],
    )

