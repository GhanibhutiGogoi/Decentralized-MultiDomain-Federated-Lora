"""Regression tests for Experiment 2 measurement-level provenance guards."""

from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT2_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT2_ROOT / "experiment"
for path in (PROJECT2_ROOT, EXPERIMENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment2.lambda_calibration import prepare_measurements
from experiment2.provenance import (
    MeasurementProvenanceError,
    load_json_without_duplicate_keys,
    normalize_measurement_provenance,
    parse_provenance_value,
    validate_experiment1_measurement_inputs,
)


def _measurements(values=None) -> pd.DataFrame:
    values = values if values is not None else [False, "False"]
    return pd.DataFrame(
        {
            "task": ["TaskA", "TaskB"][: len(values)],
            "round": [1, 1][: len(values)],
            "client_id": [0, 1][: len(values)],
            "is_synthetic": values,
            "quality_score": [0.8, 0.7][: len(values)],
            "delta_accuracy": [1.0, 0.5][: len(values)],
            "js_to_global": [0.1, 0.2][: len(values)],
            "update_l2_distance_to_mean": [2.0, 3.0][: len(values)],
            "update_cosine_distance_to_mean": [0.05, 0.10][: len(values)],
            "normalized_entropy": [0.9, 0.8][: len(values)],
            "class_imbalance_ratio": [1.2, 1.5][: len(values)],
        }
    )


def _label_distribution(values=None) -> pd.DataFrame:
    values = values if values is not None else [False, "False"]
    return pd.DataFrame(
        {
            "task": ["TaskA", "TaskB"][: len(values)],
            "client_id": [0, 1][: len(values)],
            "is_synthetic": values,
            "num_samples": [10, 20][: len(values)],
        }
    )


def _correlations(values=None) -> pd.DataFrame:
    values = values if values is not None else [False]
    return pd.DataFrame(
        {
            "predictor": ["js_to_global"][: len(values)],
            "is_synthetic": values,
            "pearson": [0.1][: len(values)],
            "spearman": [0.1][: len(values)],
            "n": [2][: len(values)],
        }
    )


def _regressions(values=None) -> pd.DataFrame:
    values = values if values is not None else [False]
    return pd.DataFrame(
        {
            "model_predictor": ["js_to_global"][: len(values)],
            "term": ["js_to_global"][: len(values)],
            "is_synthetic": values,
            "standardized_beta": [0.1][: len(values)],
            "standard_error": [0.01][: len(values)],
            "r_squared": [0.2][: len(values)],
            "n": [2][: len(values)],
        }
    )


def _manifest(synthetic_by_task=None) -> dict:
    synthetic_by_task = synthetic_by_task or {"TaskA": False, "TaskB": False}
    return {
        "datasets": {
            task: {"synthetic": synthetic}
            for task, synthetic in synthetic_by_task.items()
        }
    }


def _validate_all(**overrides):
    tables = {
        "measurements": _measurements(),
        "label_distribution": _label_distribution(),
        "correlations": _correlations(),
        "regressions": _regressions(),
        "dataset_manifest": _manifest(),
    }
    tables.update(overrides)
    return validate_experiment1_measurement_inputs(**tables)


def _parse_project2_file(relative_path: str) -> ast.Module:
    return ast.parse((PROJECT2_ROOT / relative_path).read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function {name!r} not found")


def _call_lines(function: ast.FunctionDef, call_name: str) -> list[int]:
    lines = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == call_name:
            lines.append(node.lineno)
        elif isinstance(func, ast.Attribute) and func.attr == call_name:
            lines.append(node.lineno)
    return sorted(lines)


class MeasurementProvenanceTest(unittest.TestCase):
    def test_all_real_measurement_rows_are_accepted(self):
        normalized = _validate_all()

        self.assertFalse(normalized["measurements"]["is_synthetic"].any())
        self.assertFalse(normalized["label_distribution"]["is_synthetic"].any())
        self.assertFalse(normalized["correlations"]["is_synthetic"].any())
        self.assertFalse(normalized["regressions"]["is_synthetic"].any())

    def test_one_synthetic_row_among_real_rows_is_rejected_with_task_and_row(self):
        with self.assertRaisesRegex(
            MeasurementProvenanceError,
            "csv_row=3 task='TaskB'",
        ):
            _validate_all(measurements=_measurements([False, True]))

    def test_all_synthetic_measurements_are_rejected(self):
        with self.assertRaisesRegex(MeasurementProvenanceError, "synthetic"):
            _validate_all(measurements=_measurements([True, True]))

    def test_mixed_real_and_synthetic_measurements_are_rejected(self):
        with self.assertRaisesRegex(MeasurementProvenanceError, "synthetic"):
            _validate_all(label_distribution=_label_distribution(["False", "True"]))

    def test_missing_provenance_column_is_rejected(self):
        measurements = _measurements().drop(columns=["is_synthetic"])

        with self.assertRaisesRegex(MeasurementProvenanceError, "missing required"):
            _validate_all(measurements=measurements)

    def test_null_and_nan_row_provenance_are_rejected(self):
        for value in [None, np.nan, pd.NA]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(MeasurementProvenanceError, "ambiguous"):
                    _validate_all(measurements=_measurements([False, value]))

    def test_malformed_or_ambiguous_provenance_values_are_rejected(self):
        for value in ["yes", "no", "", "Falsey", 2, -1, 0.5]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(MeasurementProvenanceError, "ambiguous"):
                    _validate_all(measurements=_measurements([False, value]))

    def test_valid_serialized_false_values_are_parsed_correctly(self):
        valid_false_values = [False, np.bool_(False), "False", "false", " 0 ", "  false  ", 0, 0.0]
        df = pd.DataFrame({"is_synthetic": valid_false_values})

        normalized = normalize_measurement_provenance(df, file_label="fixture.csv")

        self.assertFalse(normalized["is_synthetic"].any())

    def test_strings_are_not_validated_using_generic_truthiness(self):
        self.assertIs(parse_provenance_value("False"), False)
        self.assertIs(parse_provenance_value("false"), False)
        self.assertIsNone(parse_provenance_value("Falsey"))

    def test_measurement_manifest_provenance_disagreement_is_rejected(self):
        with self.assertRaisesRegex(MeasurementProvenanceError, "disagrees"):
            _validate_all(
                measurements=_measurements([False, False]),
                dataset_manifest=_manifest({"TaskA": False, "TaskB": True}),
            )

    def test_measurement_task_missing_from_manifest_is_rejected(self):
        with self.assertRaisesRegex(MeasurementProvenanceError, "missing from"):
            _validate_all(dataset_manifest=_manifest({"TaskA": False}))

    def test_manifest_task_missing_from_per_row_measurements_is_rejected(self):
        with self.assertRaisesRegex(MeasurementProvenanceError, "missing measurement rows"):
            _validate_all(measurements=_measurements([False]))

    def test_missing_task_identifier_is_rejected(self):
        measurements = _measurements([False, False])
        measurements.loc[1, "task"] = np.nan

        with self.assertRaisesRegex(MeasurementProvenanceError, "missing task"):
            _validate_all(measurements=measurements)

    def test_manifest_missing_malformed_or_synthetic_provenance_is_rejected(self):
        bad_manifests = [
            {"datasets": {"TaskA": {}, "TaskB": {"synthetic": False}}},
            {"datasets": {"TaskA": {"synthetic": "unknown"}, "TaskB": {"synthetic": False}}},
            {"datasets": {"TaskA": {"synthetic": False}, "TaskB": {"synthetic": True}}},
        ]
        for manifest in bad_manifests:
            with self.subTest(manifest=manifest):
                with self.assertRaises(MeasurementProvenanceError):
                    _validate_all(dataset_manifest=manifest)

    def test_duplicate_conflicting_manifest_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset_manifest.json"
            path.write_text(
                '{"datasets":{"TaskA":{"synthetic":false},"TaskA":{"synthetic":true}}}',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(MeasurementProvenanceError, "duplicate JSON key"):
                load_json_without_duplicate_keys(path)

    def test_every_consumed_measurement_file_is_validated(self):
        bad_regressions = _regressions().drop(columns=["is_synthetic"])

        with self.assertRaisesRegex(MeasurementProvenanceError, "controlled_regression"):
            _validate_all(
                regressions=bad_regressions,
                regressions_label="controlled_regression.csv",
            )

    def test_prepare_measurements_rejects_synthetic_before_feature_creation(self):
        with self.assertRaises(MeasurementProvenanceError):
            prepare_measurements(_measurements([False, True]))

    def test_prepare_measurements_accepts_existing_valid_real_schema(self):
        prepared = prepare_measurements(_measurements(["False", "0"]))

        self.assertIn("log_update_l2", prepared.columns)
        self.assertIn("log_class_imbalance_ratio", prepared.columns)
        self.assertFalse(prepared["is_synthetic"].any())

    def test_rejection_produces_no_experiment2_scientific_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "exp2"

            with self.assertRaises(MeasurementProvenanceError):
                _validate_all(measurements=_measurements([False, True]))

            self.assertFalse(output_dir.exists())

    def test_run_validates_provenance_before_computation_and_output_writes(self):
        tree = _parse_project2_file("experiment/experiment2/run.py")
        main = _function(tree, "main")

        validation_line = _call_lines(main, "validate_experiment1_measurement_inputs")[0]
        exp2_manifest_line = _call_lines(main, "_write_experiment2_dataset_manifest")[0]
        prepare_line = _call_lines(main, "prepare_measurements")[0]
        alpha_search_line = _call_lines(main, "leave_one_task_out_evaluation")[0]
        fit_line = _call_lines(main, "fit_form_a")[0]
        output_write_line = _call_lines(main, "to_csv")[0]
        report_line = _call_lines(main, "build_evaluation_report")[0]

        self.assertLess(validation_line, exp2_manifest_line)
        self.assertLess(validation_line, prepare_line)
        self.assertLess(validation_line, alpha_search_line)
        self.assertLess(validation_line, fit_line)
        self.assertLess(validation_line, output_write_line)
        self.assertLess(validation_line, report_line)


if __name__ == "__main__":
    unittest.main()
