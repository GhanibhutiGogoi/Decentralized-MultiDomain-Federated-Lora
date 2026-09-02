"""Measurement-level provenance validation for Experiment 2 inputs."""

from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


PROVENANCE_COLUMN = "is_synthetic"


class MeasurementProvenanceError(ValueError):
    """Raised when Experiment 1 measurement provenance cannot prove real data."""


def load_json_without_duplicate_keys(path: Path) -> dict:
    """Load JSON while rejecting duplicate keys before they can be collapsed."""

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict:
        seen = set()
        out = {}
        for key, value in pairs:
            if key in seen:
                raise MeasurementProvenanceError(
                    f"{path} contains duplicate JSON key {key!r}."
                )
            seen.add(key)
            out[key] = value
        return out

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
    )


def _format_row(index: object) -> str:
    try:
        return f"csv_row={int(index) + 2}"
    except (TypeError, ValueError):
        return f"row={index}"


def _format_rows(df: pd.DataFrame, indices: list[object], limit: int = 8) -> str:
    parts = []
    for index in indices[:limit]:
        detail = _format_row(index)
        if "task" in df.columns:
            detail += f" task={df.at[index, 'task']}"
        parts.append(detail)
    suffix = "" if len(indices) <= limit else f" ... (+{len(indices) - limit} more)"
    return ", ".join(parts) + suffix


def parse_provenance_value(value: object) -> bool | None:
    """Parse supported serialized boolean provenance values.

    ``None`` means the value is missing or ambiguous and must be rejected by
    callers. This intentionally avoids unsafe truthiness such as bool("False").
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true" or normalized == "1":
            return True
        if normalized == "false" or normalized == "0":
            return False
        return None
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        if float(value) == 0.0:
            return False
        if float(value) == 1.0:
            return True
    return None


def normalize_measurement_provenance(
    df: pd.DataFrame,
    *,
    file_label: str | Path,
    reject_synthetic: bool = True,
) -> pd.DataFrame:
    """Return a copy with explicit boolean provenance or fail closed."""
    label = str(file_label)
    if PROVENANCE_COLUMN not in df.columns:
        raise MeasurementProvenanceError(
            f"{label} is missing required measurement provenance column "
            f"'{PROVENANCE_COLUMN}'. Refusing to assume unknown data are real."
        )

    out = df.copy()
    parsed = out[PROVENANCE_COLUMN].map(parse_provenance_value)
    malformed = parsed.isna()
    if malformed.any():
        bad_indices = list(out.index[malformed])
        bad_values = sorted(
            {repr(value) for value in out.loc[malformed, PROVENANCE_COLUMN].unique()}
        )
        raise MeasurementProvenanceError(
            f"{label} contains missing, null, or ambiguous "
            f"'{PROVENANCE_COLUMN}' values at {_format_rows(out, bad_indices)}. "
            f"Bad values: {bad_values}."
        )

    out[PROVENANCE_COLUMN] = parsed.astype(bool)
    synthetic = out[PROVENANCE_COLUMN]
    if reject_synthetic and synthetic.any():
        bad_indices = list(out.index[synthetic])
        raise MeasurementProvenanceError(
            f"{label} contains synthetic measurement rows at "
            f"{_format_rows(out, bad_indices)}. Experiment 2 scientific "
            "calibration requires real-data measurements."
        )
    return out


def manifest_task_provenance(dataset_manifest: Mapping[str, object]) -> dict[str, bool]:
    """Return task -> synthetic provenance from an Experiment 1 dataset manifest."""
    datasets = dataset_manifest.get("datasets")
    if not isinstance(datasets, Mapping) or not datasets:
        raise MeasurementProvenanceError(
            "Experiment 1 dataset manifest is missing a non-empty 'datasets' mapping."
        )

    task_to_synthetic: dict[str, bool] = {}
    for task, record in datasets.items():
        if not isinstance(record, Mapping):
            raise MeasurementProvenanceError(
                f"Experiment 1 dataset manifest record for task {task!r} is not an object."
            )
        if PROVENANCE_COLUMN not in record and "synthetic" not in record:
            raise MeasurementProvenanceError(
                f"Experiment 1 dataset manifest record for task {task!r} is missing "
                "required 'synthetic' provenance."
            )
        raw_value = record.get("synthetic", record.get(PROVENANCE_COLUMN))
        parsed = parse_provenance_value(raw_value)
        if parsed is None:
            raise MeasurementProvenanceError(
                f"Experiment 1 dataset manifest record for task {task!r} has "
                f"ambiguous synthetic provenance value: {raw_value!r}."
            )
        task_to_synthetic[str(task)] = parsed
    return task_to_synthetic


def validate_measurement_manifest_consistency(
    df: pd.DataFrame,
    *,
    file_label: str | Path,
    manifest_synthetic_by_task: Mapping[str, bool],
) -> None:
    """Validate row-level provenance against task-level manifest provenance."""
    label = str(file_label)
    if "task" not in df.columns:
        return

    missing_task_rows = list(df.index[df["task"].isna()])
    if missing_task_rows:
        raise MeasurementProvenanceError(
            f"{label} contains missing task identifiers at "
            f"{_format_rows(df, missing_task_rows)}."
        )

    measurement_tasks = {str(task) for task in df["task"].unique()}
    manifest_tasks = set(manifest_synthetic_by_task)
    unknown_tasks = sorted(measurement_tasks - manifest_tasks)
    if unknown_tasks:
        raise MeasurementProvenanceError(
            f"{label} contains task(s) missing from the Experiment 1 dataset "
            f"manifest: {unknown_tasks}."
        )
    missing_tasks = sorted(manifest_tasks - measurement_tasks)
    if missing_tasks:
        raise MeasurementProvenanceError(
            f"{label} is missing measurement rows for task(s) listed in the "
            f"Experiment 1 dataset manifest: {missing_tasks}."
        )

    mismatched_indices: list[object] = []
    for index, row in df.iterrows():
        task = str(row["task"])
        expected = manifest_synthetic_by_task[task]
        actual = bool(row[PROVENANCE_COLUMN])
        if actual != expected:
            mismatched_indices.append(index)
    if mismatched_indices:
        raise MeasurementProvenanceError(
            f"{label} measurement provenance disagrees with the Experiment 1 "
            f"dataset manifest at {_format_rows(df, mismatched_indices)}."
        )


def validate_experiment1_measurement_inputs(
    *,
    measurements: pd.DataFrame,
    label_distribution: pd.DataFrame,
    correlations: pd.DataFrame,
    regressions: pd.DataFrame,
    dataset_manifest: Mapping[str, object],
    measurement_label: str | Path = "per_round_client_measurements.csv",
    label_distribution_label: str | Path = "label_distribution_summary.csv",
    correlations_label: str | Path = "signal_contribution_correlations.csv",
    regressions_label: str | Path = "controlled_regression.csv",
) -> dict[str, pd.DataFrame]:
    """Validate every Experiment 1 artifact consumed by Experiment 2."""
    manifest_synthetic_by_task = manifest_task_provenance(dataset_manifest)

    normalized = {
        "measurements": normalize_measurement_provenance(
            measurements,
            file_label=measurement_label,
        ),
        "label_distribution": normalize_measurement_provenance(
            label_distribution,
            file_label=label_distribution_label,
        ),
        "correlations": normalize_measurement_provenance(
            correlations,
            file_label=correlations_label,
        ),
        "regressions": normalize_measurement_provenance(
            regressions,
            file_label=regressions_label,
        ),
    }

    for label, table in [
        (measurement_label, normalized["measurements"]),
        (label_distribution_label, normalized["label_distribution"]),
    ]:
        validate_measurement_manifest_consistency(
            table,
            file_label=label,
            manifest_synthetic_by_task=manifest_synthetic_by_task,
        )

    synthetic_manifest_tasks = sorted(
        task for task, synthetic in manifest_synthetic_by_task.items() if synthetic
    )
    if synthetic_manifest_tasks:
        raise MeasurementProvenanceError(
            "Experiment 1 dataset manifest identifies synthetic source data for "
            f"task(s): {synthetic_manifest_tasks}. Experiment 2 scientific "
            "calibration requires real-data measurements."
        )

    return normalized
