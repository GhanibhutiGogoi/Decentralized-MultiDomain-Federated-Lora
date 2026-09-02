"""Regression tests for strict Experiment 2 numeric input validation."""

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
from experiment2.numeric_validation import (
    CORRELATION_SCHEMA,
    MEASUREMENT_SCHEMA,
    MeasurementNumericValidationError,
    validate_experiment1_numeric_inputs,
    validate_numeric_table,
)


def _measurements() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "task": ["TaskA", "TaskA", "TaskB"],
            "round": [1, 1, 1],
            "client_id": [0, 1, 0],
            "is_synthetic": [False, False, False],
            "quality_score": [0.8, 0.7, 0.6],
            "delta_accuracy": [1.0, 0.5, -0.25],
            "js_to_global": [0.1, 0.2, 0.3],
            "update_l2_distance_to_mean": [2.0, 3.0, 4.0],
            "update_cosine_distance_to_mean": [0.05, 0.10, 0.15],
            "normalized_entropy": [0.9, 0.8, 0.7],
            "class_imbalance_ratio": [1.2, 1.5, 2.0],
        }
    )


def _label_distribution() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "task": ["TaskA", "TaskA", "TaskB"],
            "client_id": [0, 1, 0],
            "is_synthetic": [False, False, False],
            "num_samples": [10, 20, 30],
            "entropy": [0.5, 0.6, 0.7],
            "normalized_entropy": [0.8, 0.85, 0.9],
            "class_imbalance_ratio": [1.1, 1.2, 1.3],
            "kl_to_global": [0.01, 0.02, 0.03],
            "js_to_global": [0.04, 0.05, 0.06],
            "zero_class_count": [0, 1, 2],
        }
    )


def _correlations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "predictor": ["js_to_global", "update_l2_distance_to_mean"],
            "is_synthetic": [False, False],
            "pearson": [0.1, -0.2],
            "spearman": [0.3, -0.4],
            "n": [3, 3],
        }
    )


def _regressions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_predictor": ["js_to_global", "js_to_global"],
            "term": ["js_to_global", "local_loss"],
            "is_synthetic": [False, False],
            "standardized_beta": [0.1, -0.2],
            "standard_error": [0.01, 0.02],
            "r_squared": [0.25, 0.25],
            "n": [3, 3],
        }
    )


def _validate_all(**overrides):
    tables = {
        "measurements": _measurements(),
        "label_distribution": _label_distribution(),
        "correlations": _correlations(),
        "regressions": _regressions(),
    }
    tables.update(overrides)
    return validate_experiment1_numeric_inputs(**tables)


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


class MeasurementNumericValidationTest(unittest.TestCase):
    def test_fully_valid_finite_measurement_table_is_accepted(self):
        out = validate_numeric_table(_measurements(), MEASUREMENT_SCHEMA)

        self.assertEqual(len(out), 3)
        self.assertListEqual(list(out["task"]), ["TaskA", "TaskA", "TaskB"])

    def test_valid_input_row_order_count_and_values_are_unchanged(self):
        df = _measurements()
        out = validate_numeric_table(df, MEASUREMENT_SCHEMA)

        self.assertEqual(len(out), len(df))
        self.assertListEqual(list(out["client_id"]), [0, 1, 0])
        np.testing.assert_allclose(out["delta_accuracy"], df["delta_accuracy"])
        np.testing.assert_allclose(out["quality_score"], df["quality_score"])

    def test_valid_serialized_numeric_strings_are_accepted(self):
        df = _measurements()
        df["quality_score"] = ["0.8", "0.7", "0.6"]

        out = validate_numeric_table(df, MEASUREMENT_SCHEMA)

        np.testing.assert_allclose(out["quality_score"], [0.8, 0.7, 0.6])

    def test_missing_required_column_is_rejected(self):
        df = _measurements().drop(columns=["delta_accuracy"])

        with self.assertRaisesRegex(MeasurementNumericValidationError, "delta_accuracy"):
            validate_numeric_table(df, MEASUREMENT_SCHEMA)

    def test_none_is_rejected(self):
        self._assert_invalid_value_rejected(None, "missing")

    def test_pandas_na_is_rejected(self):
        self._assert_invalid_value_rejected(pd.NA, "missing")

    def test_nan_is_rejected(self):
        self._assert_invalid_value_rejected(np.nan, "NaN")

    def test_positive_inf_is_rejected(self):
        self._assert_invalid_value_rejected(np.inf, "non-finite")

    def test_negative_inf_is_rejected(self):
        self._assert_invalid_value_rejected(-np.inf, "non-finite")

    def test_malformed_string_is_rejected(self):
        self._assert_invalid_value_rejected("not-a-number", "malformed")

    def test_empty_string_is_rejected(self):
        self._assert_invalid_value_rejected("", "empty")

    def test_whitespace_only_string_is_rejected(self):
        self._assert_invalid_value_rejected("   ", "empty")

    def test_python_bool_is_rejected_in_numeric_columns(self):
        self._assert_invalid_value_rejected(True, "boolean")
        self._assert_invalid_value_rejected(False, "boolean")

    def test_numpy_bool_is_rejected_in_numeric_columns(self):
        self._assert_invalid_value_rejected(np.bool_(True), "boolean")
        self._assert_invalid_value_rejected(np.bool_(False), "boolean")

    def test_one_invalid_row_rejects_entire_artifact_instead_of_dropping_row(self):
        df = _measurements()
        df.loc[1, "js_to_global"] = np.nan

        with self.assertRaisesRegex(MeasurementNumericValidationError, "csv_row=3"):
            validate_numeric_table(df, MEASUREMENT_SCHEMA)

    def test_three_rows_with_one_nan_does_not_return_two_rows(self):
        df = _measurements()
        df.loc[2, "update_l2_distance_to_mean"] = np.nan

        with self.assertRaises(MeasurementNumericValidationError):
            prepare_measurements(df)

    def test_finite_row_plus_inf_row_rejects_before_feature_matrix(self):
        df = _measurements()
        df.loc[2, "update_l2_distance_to_mean"] = np.inf

        with self.assertRaisesRegex(MeasurementNumericValidationError, "non-finite"):
            prepare_measurements(df)

    def test_multiple_invalid_values_report_total_count_and_context(self):
        df = _measurements()
        df.loc[1, "quality_score"] = np.nan
        df.loc[2, "delta_accuracy"] = np.inf

        with self.assertRaises(MeasurementNumericValidationError) as caught:
            validate_numeric_table(df, MEASUREMENT_SCHEMA)

        message = str(caught.exception)
        self.assertIn("2 invalid", message)
        self.assertIn("quality_score", message)
        self.assertIn("delta_accuracy", message)
        self.assertIn("task='TaskA'", message)
        self.assertIn("client_id=1", message)
        self.assertIn("round=1", message)

    def test_multiple_consumed_artifacts_are_validated_by_actual_schema(self):
        correlations = _correlations()
        correlations["n"] = correlations["n"].astype(object)
        correlations.loc[1, "n"] = "bad"
        regressions = _regressions()
        regressions.loc[0, "standard_error"] = np.nan

        with self.assertRaisesRegex(MeasurementNumericValidationError, "correlations.csv"):
            _validate_all(
                correlations=correlations,
                correlations_label="correlations.csv",
            )
        with self.assertRaisesRegex(MeasurementNumericValidationError, "regressions.csv"):
            _validate_all(
                regressions=regressions,
                regressions_label="regressions.csv",
            )

    def test_invalid_label_distribution_numeric_value_is_rejected(self):
        label_distribution = _label_distribution()
        label_distribution["num_samples"] = label_distribution["num_samples"].astype(object)
        label_distribution.loc[0, "num_samples"] = "unknown"

        with self.assertRaisesRegex(MeasurementNumericValidationError, "label_distribution"):
            _validate_all(
                label_distribution=label_distribution,
                label_distribution_label="label_distribution_summary.csv",
            )

    def test_invalid_numeric_input_prevents_feature_preparation(self):
        df = _measurements()
        df["class_imbalance_ratio"] = df["class_imbalance_ratio"].astype(object)
        df.loc[1, "class_imbalance_ratio"] = "bad"

        with self.assertRaises(MeasurementNumericValidationError):
            prepare_measurements(df)

    def test_invalid_numeric_with_overwrite_leaves_existing_output_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "exp2"
            output_dir.mkdir()
            existing = output_dir / "lambda_values.csv"
            existing.write_text("old scientific output", encoding="utf-8")
            bad_measurements = _measurements()
            bad_measurements.loc[1, "quality_score"] = np.nan

            with self.assertRaises(MeasurementNumericValidationError):
                validate_numeric_table(bad_measurements, MEASUREMENT_SCHEMA)

            self.assertEqual(existing.read_text(encoding="utf-8"), "old scientific output")

    def test_invalid_input_creates_no_new_scientific_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "exp2"
            bad_measurements = _measurements()
            bad_measurements.loc[0, "delta_accuracy"] = np.inf

            with self.assertRaises(MeasurementNumericValidationError):
                validate_numeric_table(bad_measurements, MEASUREMENT_SCHEMA)

            self.assertFalse(output_dir.exists())

    def test_existing_provenance_validation_still_runs_in_prepare_measurements(self):
        df = _measurements()
        df.loc[0, "is_synthetic"] = True

        with self.assertRaisesRegex(Exception, "synthetic"):
            prepare_measurements(df)

    def test_run_validates_numeric_inputs_before_cleanup_and_computation(self):
        tree = _parse_project2_file("experiment/experiment2/run.py")
        main = _function(tree, "main")

        provenance_line = _call_lines(main, "validate_experiment1_measurement_inputs")[0]
        numeric_line = _call_lines(main, "validate_experiment1_numeric_inputs")[0]
        cleanup_line = _call_lines(main, "prepare_output_directory")[0]
        manifest_line = _call_lines(main, "_write_experiment2_dataset_manifest")[0]
        prepare_line = _call_lines(main, "prepare_measurements")[0]
        alpha_line = _call_lines(main, "leave_one_task_out_evaluation")[0]
        fit_line = _call_lines(main, "fit_form_a")[0]
        gamma_line = _call_lines(main, "calibrate_lambda_scales")[0]
        csv_line = _call_lines(main, "to_csv")[0]
        figure_line = _call_lines(main, "save_figures")[0]
        report_line = _call_lines(main, "build_evaluation_report")[0]

        self.assertLess(provenance_line, numeric_line)
        self.assertLess(numeric_line, cleanup_line)
        self.assertLess(cleanup_line, manifest_line)
        self.assertLess(numeric_line, prepare_line)
        self.assertLess(numeric_line, alpha_line)
        self.assertLess(numeric_line, fit_line)
        self.assertLess(numeric_line, gamma_line)
        self.assertLess(numeric_line, csv_line)
        self.assertLess(numeric_line, figure_line)
        self.assertLess(numeric_line, report_line)

    def test_first_row_only_validation_would_miss_later_invalid_rows(self):
        df = _measurements()
        df.loc[2, "quality_score"] = np.nan

        with self.assertRaisesRegex(MeasurementNumericValidationError, "csv_row=4"):
            validate_numeric_table(df, MEASUREMENT_SCHEMA)

    def test_required_value_validation_covers_report_only_tables(self):
        correlations = _correlations()
        correlations["pearson"] = correlations["pearson"].astype(object)
        correlations.loc[0, "pearson"] = ""

        with self.assertRaisesRegex(MeasurementNumericValidationError, "pearson"):
            validate_numeric_table(correlations, CORRELATION_SCHEMA)

    def _assert_invalid_value_rejected(self, value, reason_pattern: str):
        df = _measurements()
        df["quality_score"] = df["quality_score"].astype(object)
        df.loc[1, "quality_score"] = value

        with self.assertRaisesRegex(MeasurementNumericValidationError, reason_pattern):
            validate_numeric_table(df, MEASUREMENT_SCHEMA)


if __name__ == "__main__":
    unittest.main()
