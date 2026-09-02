"""Regression tests for Project 2 experiment output-directory safety."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
import ast

PROJECT2_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT2_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT2_ROOT))
UTILS_ROOT = PROJECT2_ROOT / "framework" / "utils"
if str(UTILS_ROOT) not in sys.path:
    sys.path.insert(0, str(UTILS_ROOT))

from output_safety import (
    OutputDirectorySafetyError,
    ensure_cleanup_within_allowed_root,
    ensure_disjoint_directory,
    ensure_not_cleanup_parent,
    prepare_output_directory,
)


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


def _parser_has_store_true_flag(tree: ast.Module, flag: str) -> bool:
    parse_args = _function(tree, "parse_args")
    for node in ast.walk(parse_args):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value != flag:
            continue
        for keyword in node.keywords:
            if keyword.arg == "action" and isinstance(keyword.value, ast.Constant):
                return keyword.value.value == "store_true"
    return False


def _call_has_keyword(function: ast.FunctionDef, call_name: str, keyword_name: str) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_match = (
            (isinstance(func, ast.Name) and func.id == call_name)
            or (isinstance(func, ast.Attribute) and func.attr == call_name)
        )
        if is_match and any(keyword.arg == keyword_name for keyword in node.keywords):
            return True
    return False


def _project_layout(tmp: str):
    repo_root = Path(tmp) / "repo"
    project_root = repo_root / "project-2-domain-aware-allocation"
    outputs_root = project_root / "outputs"
    exp1_root = outputs_root / "exp1"
    exp2_root = outputs_root / "exp2"
    exp1_root.mkdir(parents=True)
    exp2_root.mkdir()
    return repo_root, project_root, outputs_root, exp1_root, exp2_root


def _prepare_with_allowed(
    output_dir: Path,
    allowed_root: Path,
    repo_root: Path,
    project_root: Path,
    outputs_root: Path,
    *,
    overwrite: bool = True,
) -> Path:
    return prepare_output_directory(
        output_dir,
        overwrite=overwrite,
        experiment_name="Experiment Test",
        allowed_cleanup_root=allowed_root,
        repository_root=repo_root,
        project_root=project_root,
        shared_outputs_root=outputs_root,
    )


class OutputDirectorySafetyTest(unittest.TestCase):
    def test_nonexistent_output_directory_is_created_and_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "new-exp"

            result = prepare_output_directory(
                output_dir,
                experiment_name="Experiment Test",
            )

            self.assertEqual(result, output_dir)
            self.assertTrue(output_dir.is_dir())
            self.assertEqual(list(output_dir.iterdir()), [])

    def test_empty_output_directory_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "empty-exp"
            output_dir.mkdir()

            result = prepare_output_directory(
                output_dir,
                experiment_name="Experiment Test",
            )

            self.assertEqual(result, output_dir)
            self.assertTrue(output_dir.is_dir())
            self.assertEqual(list(output_dir.iterdir()), [])

    def test_non_empty_output_directory_is_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "exp"
            output_dir.mkdir()
            existing_file = output_dir / "manifest.json"
            existing_file.write_text("previous run", encoding="utf-8")
            existing_subdir = output_dir / "figures"
            existing_subdir.mkdir()
            nested_file = existing_subdir / "plot.svg"
            nested_file.write_text("old figure", encoding="utf-8")

            with self.assertRaises(OutputDirectorySafetyError):
                prepare_output_directory(
                    output_dir,
                    experiment_name="Experiment Test",
                )

            self.assertEqual(existing_file.read_text(encoding="utf-8"), "previous run")
            self.assertEqual(nested_file.read_text(encoding="utf-8"), "old figure")

    def test_overwrite_cleans_only_targeted_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root, project_root, outputs_root, exp1_root, exp2_root = _project_layout(
                tmp
            )
            output_dir = exp2_root
            (output_dir / "lambda_values.csv").write_text("old", encoding="utf-8")
            figures = output_dir / "figures"
            figures.mkdir()
            (figures / "lambda_distribution.svg").write_text("old", encoding="utf-8")

            sibling_file = outputs_root / "exp1_measurements.csv"
            sibling_file.write_text("must stay", encoding="utf-8")
            (exp1_root / "per_round_client_measurements.csv").write_text(
                "must stay",
                encoding="utf-8",
            )

            _prepare_with_allowed(
                output_dir,
                exp2_root,
                repo_root,
                project_root,
                outputs_root,
            )

            self.assertTrue(output_dir.is_dir())
            self.assertEqual(list(output_dir.iterdir()), [])
            self.assertEqual(sibling_file.read_text(encoding="utf-8"), "must stay")
            self.assertEqual(
                (exp1_root / "per_round_client_measurements.csv").read_text(
                    encoding="utf-8"
                ),
                "must stay",
            )

    def test_overwrite_requires_explicit_allowed_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "unscoped"
            output_dir.mkdir()
            sentinel = output_dir / "sentinel.txt"
            sentinel.write_text("must stay", encoding="utf-8")

            with self.assertRaises(OutputDirectorySafetyError):
                prepare_output_directory(
                    output_dir,
                    overwrite=True,
                    experiment_name="Experiment Test",
                )

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "must stay")

    def test_cleanup_confinement_accepts_default_roots_and_descendants(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root, project_root, outputs_root, exp1_root, exp2_root = _project_layout(
                tmp
            )
            for allowed_root in (exp1_root, exp2_root):
                with self.subTest(allowed_root=allowed_root):
                    nested = allowed_root / "rerun"
                    _prepare_with_allowed(
                        allowed_root,
                        allowed_root,
                        repo_root,
                        project_root,
                        outputs_root,
                    )
                    _prepare_with_allowed(
                        nested,
                        allowed_root,
                        repo_root,
                        project_root,
                        outputs_root,
                    )
                    self.assertTrue(nested.is_dir())

    def test_arbitrary_outside_cleanup_is_rejected_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root, project_root, outputs_root, exp1_root, _ = _project_layout(tmp)
            outside = Path(tmp) / "outside-custom"
            outside.mkdir()
            sentinel = outside / "sentinel.txt"
            sentinel.write_text("must stay", encoding="utf-8")

            with self.assertRaises(OutputDirectorySafetyError):
                _prepare_with_allowed(
                    outside,
                    exp1_root,
                    repo_root,
                    project_root,
                    outputs_root,
                )

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "must stay")

    def test_roots_parents_and_sibling_experiments_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root, project_root, outputs_root, exp1_root, exp2_root = _project_layout(
                tmp
            )
            root_candidate = Path(exp1_root.anchor)
            forbidden = [
                repo_root,
                project_root,
                outputs_root,
                exp1_root.parent,
                exp1_root.parent.parent,
                exp2_root,
                root_candidate,
            ]
            for output_dir in forbidden:
                marker = None
                if output_dir != root_candidate:
                    marker = output_dir / "marker.txt"
                    marker.write_text("must stay", encoding="utf-8")
                with self.subTest(output_dir=output_dir):
                    with self.assertRaises(OutputDirectorySafetyError):
                        _prepare_with_allowed(
                            output_dir,
                            exp1_root,
                            repo_root,
                            project_root,
                            outputs_root,
                        )
                    if marker is not None:
                        self.assertEqual(marker.read_text(encoding="utf-8"), "must stay")

    def test_dotdot_prefix_absolute_and_relative_escapes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root, project_root, outputs_root, exp1_root, _ = _project_layout(tmp)
            dotdot_escape = exp1_root / ".." / "dotdot-escape"
            prefix_escape = outputs_root / "exp1-backup"
            absolute_escape = Path(tmp) / "absolute-outside"
            for path in (dotdot_escape, prefix_escape, absolute_escape):
                path.mkdir()
                (path / "sentinel.txt").write_text("must stay", encoding="utf-8")

            previous_cwd = Path.cwd()
            try:
                os.chdir(tmp)
                relative_escape = Path("relative-outside")
                relative_escape.mkdir()
                (relative_escape / "sentinel.txt").write_text(
                    "must stay",
                    encoding="utf-8",
                )
                candidates = [
                    dotdot_escape,
                    prefix_escape,
                    absolute_escape,
                    relative_escape,
                ]
                for output_dir in candidates:
                    with self.subTest(output_dir=output_dir):
                        with self.assertRaises(OutputDirectorySafetyError):
                            _prepare_with_allowed(
                                output_dir,
                                exp1_root,
                                repo_root,
                                project_root,
                                outputs_root,
                            )
                        self.assertEqual(
                            (Path(output_dir) / "sentinel.txt").read_text(
                                encoding="utf-8"
                            ),
                            "must stay",
                        )
            finally:
                os.chdir(previous_cwd)

    def test_file_target_is_rejected_safely(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root, project_root, outputs_root, exp1_root, _ = _project_layout(tmp)
            file_target = exp1_root / "not-a-directory.txt"
            file_target.write_text("must stay", encoding="utf-8")

            with self.assertRaises(OutputDirectorySafetyError):
                _prepare_with_allowed(
                    file_target,
                    exp1_root,
                    repo_root,
                    project_root,
                    outputs_root,
                )

            self.assertEqual(file_target.read_text(encoding="utf-8"), "must stay")

    def test_non_empty_allowed_target_without_overwrite_is_rejected_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root, project_root, outputs_root, exp1_root, _ = _project_layout(tmp)
            sentinel = exp1_root / "sentinel.txt"
            sentinel.write_text("must stay", encoding="utf-8")

            with self.assertRaises(OutputDirectorySafetyError):
                _prepare_with_allowed(
                    exp1_root,
                    exp1_root,
                    repo_root,
                    project_root,
                    outputs_root,
                    overwrite=False,
                )

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "must stay")

    def test_symlink_escape_is_rejected_when_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root, project_root, outputs_root, exp1_root, _ = _project_layout(tmp)
            outside = Path(tmp) / "outside-destination"
            outside.mkdir()
            sentinel = outside / "sentinel.txt"
            sentinel.write_text("must stay", encoding="utf-8")
            link = exp1_root / "linked-output"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"directory symlink unavailable on this platform: {exc}")

            with self.assertRaises(OutputDirectorySafetyError):
                _prepare_with_allowed(
                    link,
                    exp1_root,
                    repo_root,
                    project_root,
                    outputs_root,
                )

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "must stay")

    def test_child_symlink_destination_is_not_cleaned_when_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root, project_root, outputs_root, exp1_root, _ = _project_layout(tmp)
            outside = Path(tmp) / "outside-destination"
            outside.mkdir()
            sentinel = outside / "sentinel.txt"
            sentinel.write_text("must stay", encoding="utf-8")
            child_link = exp1_root / "child-link"
            try:
                child_link.symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"directory symlink unavailable on this platform: {exc}")

            _prepare_with_allowed(
                exp1_root,
                exp1_root,
                repo_root,
                project_root,
                outputs_root,
            )

            self.assertFalse(child_link.exists())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "must stay")

    def test_importing_output_safety_helper_has_no_cleanup_side_effect(self):
        with tempfile.TemporaryDirectory() as tmp:
            sentinel = Path(tmp) / "sentinel.txt"
            sentinel.write_text("must stay", encoding="utf-8")

            __import__("output_safety")

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "must stay")

    def test_experiment2_output_scope_cannot_overlap_experiment1_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp1_dir = root / "outputs" / "exp1"
            exp1_dir.mkdir(parents=True)
            measurement = exp1_dir / "per_round_client_measurements.csv"
            measurement.write_text("previous measurements", encoding="utf-8")

            overlapping_outputs = [
                exp1_dir,
                exp1_dir / "nested-exp2",
                exp1_dir.parent,
            ]
            for output_dir in overlapping_outputs:
                with self.subTest(output_dir=output_dir):
                    with self.assertRaises(OutputDirectorySafetyError):
                        ensure_disjoint_directory(
                            output_dir,
                            exp1_dir,
                            output_label="Experiment 2 output directory",
                            protected_label="Experiment 1 input directory",
                        )

            self.assertEqual(
                measurement.read_text(encoding="utf-8"),
                "previous measurements",
            )

    def test_experiment1_output_scope_cannot_overlap_experiment2_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs_root = root / "outputs"
            exp2_dir = outputs_root / "exp2"
            exp2_dir.mkdir(parents=True)
            exp2_manifest = exp2_dir / "manifest.json"
            exp2_manifest.write_text("experiment 2", encoding="utf-8")

            overlapping_outputs = [
                exp2_dir,
                exp2_dir / "nested-exp1",
                outputs_root,
                outputs_root / "exp1" / "..",
            ]
            for output_dir in overlapping_outputs:
                with self.subTest(output_dir=output_dir):
                    with self.assertRaises(OutputDirectorySafetyError):
                        ensure_disjoint_directory(
                            output_dir,
                            exp2_dir,
                            output_label="Experiment 1 output directory",
                            protected_label="Experiment 2 output directory",
                        )

            self.assertEqual(exp2_manifest.read_text(encoding="utf-8"), "experiment 2")

    def test_shared_outputs_parent_cannot_be_cleanup_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs_root = root / "outputs"
            exp1_dir = outputs_root / "exp1"
            exp2_dir = outputs_root / "exp2"
            exp1_dir.mkdir(parents=True)
            exp2_dir.mkdir()
            exp1_measurement = exp1_dir / "per_round_client_measurements.csv"
            exp2_manifest = exp2_dir / "manifest.json"
            exp1_measurement.write_text("experiment 1", encoding="utf-8")
            exp2_manifest.write_text("experiment 2", encoding="utf-8")

            forbidden_outputs = [
                outputs_root,
                outputs_root / "exp1" / "..",
                root,
            ]
            for output_dir in forbidden_outputs:
                with self.subTest(output_dir=output_dir):
                    with self.assertRaises(OutputDirectorySafetyError):
                        ensure_not_cleanup_parent(
                            output_dir,
                            outputs_root,
                            output_label="experiment output directory",
                            protected_label="shared Project 2 outputs directory",
                        )

            self.assertEqual(
                exp1_measurement.read_text(encoding="utf-8"),
                "experiment 1",
            )
            self.assertEqual(exp2_manifest.read_text(encoding="utf-8"), "experiment 2")

    def test_experiment1_preflight_is_before_dataset_loading_and_training(self):
        tree = _parse_project2_file("experiment/experiment1/run.py")
        main = _function(tree, "main")

        disjoint_line = _call_lines(main, "ensure_disjoint_directory")[0]
        parent_guard_line = _call_lines(main, "ensure_not_cleanup_parent")[0]
        guard_line = _call_lines(main, "prepare_output_directory")[0]
        load_line = _call_lines(main, "load_experiments")[0]
        manifest_line = _call_lines(main, "write_dataset_manifest")[0]
        run_task_line = _call_lines(main, "run_task")[0]

        self.assertTrue(
            _call_has_keyword(main, "prepare_output_directory", "allowed_cleanup_root")
        )
        self.assertLess(disjoint_line, guard_line)
        self.assertLess(parent_guard_line, guard_line)
        self.assertLess(guard_line, load_line)
        self.assertLess(guard_line, manifest_line)
        self.assertLess(guard_line, run_task_line)

    def test_experiment2_preflight_is_before_input_reads_and_writes(self):
        tree = _parse_project2_file("experiment/experiment2/run.py")
        main = _function(tree, "main")

        disjoint_line = _call_lines(main, "ensure_disjoint_directory")[0]
        parent_guard_line = _call_lines(main, "ensure_not_cleanup_parent")[0]
        guard_line = _call_lines(main, "prepare_output_directory")[0]
        first_read_line = _call_lines(main, "_read_required_csv")[0]
        provenance_line = _call_lines(main, "validate_experiment1_measurement_inputs")[0]
        numeric_line = _call_lines(main, "validate_experiment1_numeric_inputs")[0]
        manifest_write_line = _call_lines(main, "_write_experiment2_dataset_manifest")[0]

        self.assertTrue(
            _call_has_keyword(main, "prepare_output_directory", "allowed_cleanup_root")
        )
        self.assertLess(disjoint_line, first_read_line)
        self.assertLess(parent_guard_line, first_read_line)
        self.assertLess(first_read_line, provenance_line)
        self.assertLess(provenance_line, numeric_line)
        self.assertLess(numeric_line, guard_line)
        self.assertLess(guard_line, manifest_write_line)

    def test_cli_flags_remain_compatible_with_new_overwrite_flag(self):
        exp1_tree = _parse_project2_file("experiment/experiment1/run.py")
        exp2_tree = _parse_project2_file("experiment/experiment2/run.py")

        self.assertTrue(_parser_has_store_true_flag(exp1_tree, "--overwrite"))
        self.assertTrue(_parser_has_store_true_flag(exp1_tree, "--download-datasets"))
        self.assertTrue(_parser_has_store_true_flag(exp1_tree, "--pin-memory"))
        self.assertTrue(_parser_has_store_true_flag(exp2_tree, "--overwrite"))
        self.assertTrue(_parser_has_store_true_flag(exp2_tree, "--allow-synthetic-source"))
        self.assertTrue(
            _parser_has_store_true_flag(exp2_tree, "--include-extended-ridge-alphas")
        )


if __name__ == "__main__":
    unittest.main()
