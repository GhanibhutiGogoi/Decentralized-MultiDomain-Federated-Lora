"""Strict numeric validation for Experiment 2 input artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .diagnostics import format_diagnostic_value


class MeasurementNumericValidationError(ValueError):
    """Raised when required Experiment 1 numeric measurements are invalid."""


@dataclass(frozen=True)
class NumericArtifactSchema:
    """Required numeric and context columns for a consumed measurement artifact."""

    artifact_name: str
    required_numeric_columns: tuple[str, ...]
    identifier_columns: tuple[str, ...] = ()
    provenance_columns: tuple[str, ...] = ("is_synthetic",)
    optional_numeric_columns: tuple[str, ...] = ()


MEASUREMENT_SCHEMA = NumericArtifactSchema(
    artifact_name="per_round_client_measurements.csv",
    required_numeric_columns=(
        "quality_score",
        "delta_accuracy",
        "js_to_global",
        "update_l2_distance_to_mean",
        "update_cosine_distance_to_mean",
        "normalized_entropy",
        "class_imbalance_ratio",
    ),
    identifier_columns=("task", "round", "client_id"),
)


LABEL_DISTRIBUTION_SCHEMA = NumericArtifactSchema(
    artifact_name="label_distribution_summary.csv",
    required_numeric_columns=(
        "num_samples",
        "entropy",
        "normalized_entropy",
        "class_imbalance_ratio",
        "kl_to_global",
        "js_to_global",
        "zero_class_count",
    ),
    identifier_columns=("task", "client_id"),
)


CORRELATION_SCHEMA = NumericArtifactSchema(
    artifact_name="signal_contribution_correlations.csv",
    required_numeric_columns=("pearson", "spearman", "n"),
    identifier_columns=("predictor",),
)


REGRESSION_SCHEMA = NumericArtifactSchema(
    artifact_name="controlled_regression.csv",
    required_numeric_columns=("standardized_beta", "standard_error", "r_squared", "n"),
    identifier_columns=("model_predictor", "term"),
)


EXPERIMENT1_NUMERIC_SCHEMAS: Mapping[str, NumericArtifactSchema] = {
    "measurements": MEASUREMENT_SCHEMA,
    "label_distribution": LABEL_DISTRIBUTION_SCHEMA,
    "correlations": CORRELATION_SCHEMA,
    "regressions": REGRESSION_SCHEMA,
}


def _row_label(index: object) -> str:
    try:
        return f"csv_row={int(index) + 2}"
    except (TypeError, ValueError):
        return f"row={index}"


def _context(df: pd.DataFrame, index: object) -> str:
    parts = []
    for column, label in [
        ("task", "task"),
        ("round", "round"),
        ("client_id", "client_id"),
        ("predictor", "predictor"),
        ("model_predictor", "model_predictor"),
        ("term", "term"),
    ]:
        if column in df.columns:
            parts.append(f"{label}={format_diagnostic_value(df.at[index, column])}")
    return " ".join(parts)


def _invalid_message(
    artifact_name: str,
    df: pd.DataFrame,
    invalid: Sequence[tuple[object, str, object, str]],
    *,
    limit: int = 8,
) -> str:
    examples = []
    for index, column, value, reason in invalid[:limit]:
        context = _context(df, index)
        if context:
            context = " " + context
        examples.append(
            f"{_row_label(index)}{context} column={column!r} "
            f"value={format_diagnostic_value(value)} reason={reason}"
        )
    suffix = "" if len(invalid) <= limit else f" ... (+{len(invalid) - limit} more)"
    return (
        f"{artifact_name} contains {len(invalid)} invalid required numeric "
        f"value(s): {'; '.join(examples)}{suffix}"
    )


def _parse_required_numeric(value: object) -> tuple[float | None, str | None]:
    if isinstance(value, (bool, np.bool_)):
        return None, "boolean is not a scientific numeric value"
    if value is None:
        return None, "missing value"
    try:
        if pd.isna(value):
            return None, "missing/null/NaN value"
    except (TypeError, ValueError):
        return None, "ambiguous value"

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None, "empty string"
        try:
            parsed = float(stripped)
        except ValueError:
            return None, "malformed non-numeric string"
    else:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None, "malformed non-numeric value"

    if not np.isfinite(parsed):
        return None, "non-finite value"
    return parsed, None


def _validate_required_columns(df: pd.DataFrame, schema: NumericArtifactSchema) -> None:
    required = [
        *schema.identifier_columns,
        *schema.provenance_columns,
        *schema.required_numeric_columns,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise MeasurementNumericValidationError(
            f"{schema.artifact_name} is missing required column(s): {missing}."
        )


def _validate_identifier_values(df: pd.DataFrame, schema: NumericArtifactSchema) -> None:
    invalid: list[tuple[object, str, object, str]] = []
    for column in schema.identifier_columns:
        for index, value in df[column].items():
            if value is None:
                invalid.append((index, column, value, "missing identifier"))
                continue
            try:
                missing = bool(pd.isna(value))
            except (TypeError, ValueError):
                missing = False
            if missing:
                invalid.append((index, column, value, "missing identifier"))
            elif isinstance(value, str) and not value.strip():
                invalid.append((index, column, value, "empty identifier"))
    if invalid:
        raise MeasurementNumericValidationError(
            _invalid_message(schema.artifact_name, df, invalid)
        )


def validate_numeric_table(
    df: pd.DataFrame,
    schema: NumericArtifactSchema,
    *,
    artifact_label: str | Path | None = None,
) -> pd.DataFrame:
    """Validate and return a copy with required numeric columns parsed.

    No rows are dropped, imputed, clipped, normalized, or reordered.
    """
    label = str(artifact_label or schema.artifact_name)
    schema = NumericArtifactSchema(
        artifact_name=label,
        required_numeric_columns=schema.required_numeric_columns,
        identifier_columns=schema.identifier_columns,
        provenance_columns=schema.provenance_columns,
        optional_numeric_columns=schema.optional_numeric_columns,
    )
    _validate_required_columns(df, schema)
    _validate_identifier_values(df, schema)

    out = df.copy()
    invalid: list[tuple[object, str, object, str]] = []
    parsed_columns: dict[str, list[float]] = {
        column: [] for column in schema.required_numeric_columns
    }

    for index, row in out.iterrows():
        for column in schema.required_numeric_columns:
            parsed, reason = _parse_required_numeric(row[column])
            if reason is not None:
                invalid.append((index, column, row[column], reason))
            else:
                parsed_columns[column].append(float(parsed))

    if invalid:
        raise MeasurementNumericValidationError(
            _invalid_message(schema.artifact_name, out, invalid)
        )

    for column, values in parsed_columns.items():
        out[column] = values
    return out


def validate_experiment1_numeric_inputs(
    *,
    measurements: pd.DataFrame,
    label_distribution: pd.DataFrame,
    correlations: pd.DataFrame,
    regressions: pd.DataFrame,
    measurement_label: str | Path = MEASUREMENT_SCHEMA.artifact_name,
    label_distribution_label: str | Path = LABEL_DISTRIBUTION_SCHEMA.artifact_name,
    correlations_label: str | Path = CORRELATION_SCHEMA.artifact_name,
    regressions_label: str | Path = REGRESSION_SCHEMA.artifact_name,
) -> dict[str, pd.DataFrame]:
    """Validate every Experiment 1 numeric artifact consumed by Experiment 2."""
    return {
        "measurements": validate_numeric_table(
            measurements,
            MEASUREMENT_SCHEMA,
            artifact_label=measurement_label,
        ),
        "label_distribution": validate_numeric_table(
            label_distribution,
            LABEL_DISTRIBUTION_SCHEMA,
            artifact_label=label_distribution_label,
        ),
        "correlations": validate_numeric_table(
            correlations,
            CORRELATION_SCHEMA,
            artifact_label=correlations_label,
        ),
        "regressions": validate_numeric_table(
            regressions,
            REGRESSION_SCHEMA,
            artifact_label=regressions_label,
        ),
    }
