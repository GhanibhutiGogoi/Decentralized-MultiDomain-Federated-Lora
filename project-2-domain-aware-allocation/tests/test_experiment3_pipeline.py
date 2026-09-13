"""Engineering tests for the Experiment 3 pipeline scaffold."""

from __future__ import annotations

import ast
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT2_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT2_ROOT, PROJECT2_ROOT / "experiment"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment3.arms import (  # noqa: E402
    ClientRecord,
    Experiment3ArmError,
    calculate_lambda_weights,
    realized_aggregation_weights,
    validate_lambda_weights,
)
from experiment3.calibration_bundle import (  # noqa: E402
    CalibrationBundleError,
    load_calibration_bundle,
)
from experiment3.checkpoint import (  # noqa: E402
    Experiment3CheckpointError,
    RunIdentity,
    checkpoint_payload,
    read_checkpoint,
    stable_hash,
    write_checkpoint,
)
from experiment3.metrics import (  # noqa: E402
    Experiment3MetricError,
    convergence_speed,
    fairness_spread_variance,
    final_round_global_accuracy,
    global_accuracy_per_round,
    per_domain_accuracy,
    realized_client_weights,
    within_round_rank_agreement,
    worst_domain_accuracy,
)
from experiment3.partitions import (  # noqa: E402
    PartitionValidationError,
    fresh_dirichlet_partition,
    validate_partition,
)
import experiment3.run as exp3_run  # noqa: E402
from experiment3.run import (  # noqa: E402
    ClientUpdate,
    Experiment3Config,
    PreparedTask,
    TaskRuntime,
)
from experiment3.seeds import derive_seeds  # noqa: E402
from experiment3.statistics import (  # noqa: E402
    Experiment3StatisticsError,
    compare_to_mde,
    confidence_interval,
    paired_arm_differences,
    paired_permutation_test,
    pooled_summary,
    task_level_summary,
)
from framework.utils import OutputDirectorySafetyError  # noqa: E402


FIXTURE = PROJECT2_ROOT / "tests" / "fixtures" / "experiment3" / "calibration_bundle.json"


def load_fixture_bundle():
    return load_calibration_bundle(
        FIXTURE,
        expected_tasks=["TinyTask"],
        expected_experiment1_run_id="exp1-engineering-fixture",
        expected_experiment2_run_id="exp2-engineering-fixture",
        allow_engineering_fixture=True,
    )


def client_records():
    features_1 = {
        "log_update_l2": 3.0,
        "js_to_global": 0.1,
        "update_cosine_distance_to_mean": 0.5,
        "normalized_entropy": 0.6,
        "log_class_imbalance_ratio": 1.0,
    }
    features_2 = {
        "log_update_l2": 1.0,
        "js_to_global": 0.3,
        "update_cosine_distance_to_mean": 0.1,
        "normalized_entropy": 0.2,
        "log_class_imbalance_ratio": 0.5,
    }
    return [
        ClientRecord("TinyTask", 1, "client_0", 10, 1.0, features_1),
        ClientRecord("TinyTask", 1, "client_1", 30, 2.0, features_2),
    ]


class InMemoryExperiment3Ops:
    def __init__(self):
        self.aggregate_calls = []

    def load_tasks(self, config):
        return [TaskRuntime("TinyTask", {"labels": [0, 1]})]

    def prepare_task(self, task, config, seeds):
        return PreparedTask(
            partition_hash=f"partition-{seeds.partition_seed}",
            client_ids=("client_0", "client_1"),
            payload={},
        )

    def initial_state(self, task, config):
        return {"value": 0.0}

    def train_client_updates(self, task, prepared, *, arm, round_id, global_state, config):
        records = client_records()
        return [
            ClientUpdate(
                client_id=record.client_id,
                state={"delta": 1.0 + idx},
                sample_count=record.sample_count,
                quality_score=record.quality_score,
                features=record.features,
                domain=f"domain_{idx}",
                intended_domain_divergence=record.features["js_to_global"],
            )
            for idx, record in enumerate(records)
        ]

    def aggregate(
        self,
        task,
        prepared,
        *,
        arm,
        round_id,
        global_state,
        updates,
        lambda_weights,
        config,
    ):
        applied = [1.0 for _ in updates] if lambda_weights is None else list(lambda_weights)
        self.aggregate_calls.append((arm, tuple(round(float(v), 12) for v in applied)))
        increment = sum(
            update.state["delta"] * update.sample_count * update.quality_score * lam
            for update, lam in zip(updates, applied)
        )
        return {"value": global_state["value"] + increment}

    def evaluate(self, task, prepared, *, arm, round_id, global_state, config):
        arm_offset = {"baseline": 0.0, "form_a": 1.0, "form_b": 2.0}[arm]
        global_accuracy = 50.0 + arm_offset + round_id
        return global_accuracy, {
            "client_0": global_accuracy - 1.0,
            "client_1": global_accuracy - 2.0,
        }

    def save_state(self, path, state):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")

    def load_state(self, path):
        return json.loads(path.read_text(encoding="utf-8"))


def write_bundle(tmp: Path, edits: dict | None = None) -> Path:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if edits:
        for dotted_key, value in edits.items():
            target = data
            parts = dotted_key.split(".")
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
    path = tmp / "bundle.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


class CalibrationBundleTest(unittest.TestCase):
    def test_production_rejects_synthetic_test_calibration(self):
        with self.assertRaises(CalibrationBundleError):
            load_calibration_bundle(
                FIXTURE,
                expected_tasks=["TinyTask"],
                expected_experiment1_run_id="exp1-engineering-fixture",
                expected_experiment2_run_id="exp2-engineering-fixture",
            )

    def test_fixture_requires_private_injection(self):
        bundle = load_fixture_bundle()
        self.assertEqual(bundle.purpose, "engineering_test_only")
        self.assertTrue(bundle.is_synthetic)

    def test_missing_and_malformed_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_bundle(Path(tmp))
            data = json.loads(path.read_text(encoding="utf-8"))
            del data["forms"]["form_a"]["gamma"]
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(CalibrationBundleError):
                load_calibration_bundle(
                    path,
                    expected_tasks=["TinyTask"],
                    expected_experiment1_run_id="exp1-engineering-fixture",
                    expected_experiment2_run_id="exp2-engineering-fixture",
                    allow_engineering_fixture=True,
                )

    def test_non_finite_coefficients_and_stds_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_bundle(Path(tmp), {"forms.form_a.coefficients": [math.inf, 0.0]})
            with self.assertRaises(CalibrationBundleError):
                load_calibration_bundle(
                    path,
                    expected_tasks=["TinyTask"],
                    expected_experiment1_run_id="exp1-engineering-fixture",
                    expected_experiment2_run_id="exp2-engineering-fixture",
                    allow_engineering_fixture=True,
                )
            path = write_bundle(
                Path(tmp),
                {"forms.form_a.standardization_stds": {"log_update_l2": 0.0, "js_to_global": 0.1}},
            )
            with self.assertRaises(CalibrationBundleError):
                load_calibration_bundle(
                    path,
                    expected_tasks=["TinyTask"],
                    expected_experiment1_run_id="exp1-engineering-fixture",
                    expected_experiment2_run_id="exp2-engineering-fixture",
                    allow_engineering_fixture=True,
                )

    def test_task_set_and_run_identity_mismatch_are_rejected(self):
        with self.assertRaises(CalibrationBundleError):
            load_calibration_bundle(
                FIXTURE,
                expected_tasks=["OtherTask"],
                expected_experiment1_run_id="exp1-engineering-fixture",
                expected_experiment2_run_id="exp2-engineering-fixture",
                allow_engineering_fixture=True,
            )
        with self.assertRaises(CalibrationBundleError):
            load_calibration_bundle(
                FIXTURE,
                expected_tasks=["TinyTask"],
                expected_experiment1_run_id="wrong",
                expected_experiment2_run_id="exp2-engineering-fixture",
                allow_engineering_fixture=True,
            )

    def test_duplicate_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "duplicate.json"
            path.write_text('{"schema_version":"x","schema_version":"y"}', encoding="utf-8")
            with self.assertRaises(CalibrationBundleError):
                load_calibration_bundle(
                    path,
                    expected_tasks=["TinyTask"],
                    expected_experiment1_run_id="x",
                    expected_experiment2_run_id="y",
                    allow_engineering_fixture=True,
                )


class ArmMathTest(unittest.TestCase):
    def test_correct_form_a_and_form_b_calculation(self):
        bundle = load_fixture_bundle()
        records = client_records()

        form_a = calculate_lambda_weights(records, bundle, "form_a")
        raw_a = np.exp(np.array([0.4 * 0.5, 0.4 * -0.5]))
        expected_a = raw_a / raw_a.mean()
        self.assertTrue(np.allclose(form_a, expected_a, atol=1e-12))

        form_b = calculate_lambda_weights(records, bundle, "form_b")
        score_1 = -0.2 + 0.2 * 1.0 + 0.1 * -1.0 - 0.3 * 1.0 + 0.4 * 1.0 + 0.5 * 1.0
        score_2 = -0.2 + 0.2 * 0.0 + 0.1 * 1.0 - 0.3 * -1.0 + 0.4 * -1.0 + 0.5 * 0.0
        raw_b = np.exp(np.array([0.3 * ((score_1 - score_2) / 2), 0.3 * ((score_2 - score_1) / 2)]))
        expected_b = raw_b / raw_b.mean()
        self.assertTrue(np.allclose(form_b, expected_b, atol=1e-12))
        self.assertFalse(np.allclose(form_a, form_b))

    def test_exact_baseline_compatibility(self):
        records = client_records()
        baseline = realized_aggregation_weights(records, "baseline")
        self.assertEqual(baseline, [10.0 / 70.0, 60.0 / 70.0])

    def test_client_ordering_and_invalid_lambda_weights(self):
        records = client_records()
        bundle = load_fixture_bundle()
        original = calculate_lambda_weights(records, bundle, "form_a")
        reversed_weights = calculate_lambda_weights(list(reversed(records)), bundle, "form_a")
        self.assertTrue(np.allclose(original, list(reversed(reversed_weights))))
        with self.assertRaises(Experiment3ArmError):
            validate_lambda_weights(records, [1.0])
        with self.assertRaises(Experiment3ArmError):
            validate_lambda_weights(records, [1.0, math.nan])
        with self.assertRaises(Experiment3ArmError):
            validate_lambda_weights(records, [1.0, -0.1])
        with self.assertRaises(Experiment3ArmError):
            calculate_lambda_weights([records[0], records[0]], bundle, "form_a")

    def test_wrong_feature_order_would_change_result(self):
        bundle = load_fixture_bundle()
        records = client_records()
        correct = calculate_lambda_weights(records, bundle, "form_a")
        swapped_score_0 = 0.1 + 0.5 * ((0.1 - 1.0) / 2.0) + -0.25 * ((3.0 - 0.2) / 0.1)
        swapped_score_1 = 0.1 + 0.5 * ((0.3 - 1.0) / 2.0) + -0.25 * ((1.0 - 0.2) / 0.1)
        raw = np.exp(np.array([0.4 * ((swapped_score_0 - swapped_score_1) / 2), 0.4 * ((swapped_score_1 - swapped_score_0) / 2)]))
        wrong = raw / raw.mean()
        self.assertFalse(np.allclose(correct, wrong))


class SeedPartitionTest(unittest.TestCase):
    def test_deterministic_seeds(self):
        self.assertEqual(derive_seeds(7), derive_seeds(7))
        self.assertNotEqual(derive_seeds(7).partition_seed, derive_seeds(8).partition_seed)

    def test_fresh_deterministic_partitions_and_sample_conservation(self):
        labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        a = fresh_dirichlet_partition(labels, num_clients=3, alpha=0.5, seed=11)
        b = fresh_dirichlet_partition(labels, num_clients=3, alpha=0.5, seed=11)
        c = fresh_dirichlet_partition(labels, num_clients=3, alpha=0.5, seed=12)
        self.assertEqual(a.indices_by_client, b.indices_by_client)
        self.assertEqual(a.partition_hash, b.partition_hash)
        self.assertNotEqual(a.partition_hash, c.partition_hash)
        validate_partition(a.indices_by_client, len(labels))
        with self.assertRaises(PartitionValidationError):
            validate_partition([[0, 1], [1, 2]], 3)


class MetricsTest(unittest.TestCase):
    def observations(self):
        return pd.DataFrame(
            [
                ["TinyTask", 1, "baseline", 1, "c0", "d0", 0.70],
                ["TinyTask", 1, "baseline", 1, "c1", "d1", 0.50],
                ["TinyTask", 1, "baseline", 2, "c0", "d0", 0.80],
                ["TinyTask", 1, "baseline", 2, "c1", "d1", 0.60],
                ["TinyTask", 1, "form_a", 1, "c0", "d0", 0.75],
                ["TinyTask", 1, "form_a", 1, "c1", "d1", 0.55],
            ],
            columns=["task", "seed", "arm", "round", "client_id", "domain", "accuracy"],
        )

    def test_metric_correctness_and_edges(self):
        df = self.observations()
        curve = global_accuracy_per_round(df)
        self.assertAlmostEqual(
            curve[(curve["arm"] == "baseline") & (curve["round"] == 1)]["global_accuracy"].iloc[0],
            0.60,
        )
        self.assertAlmostEqual(final_round_global_accuracy(df).query("arm == 'baseline'")["global_accuracy"].iloc[0], 0.70)
        self.assertEqual(convergence_speed(df, 0.70).query("arm == 'baseline'")["round_to_target"].iloc[0], 2)
        self.assertEqual(len(per_domain_accuracy(df)), len(df))
        self.assertAlmostEqual(worst_domain_accuracy(df).query("arm == 'baseline' and round == 1")["worst_domain_accuracy"].iloc[0], 0.50)
        self.assertAlmostEqual(fairness_spread_variance(df).query("arm == 'baseline' and round == 1")["domain_accuracy_variance"].iloc[0], 0.01)

    def test_no_silent_row_dropping(self):
        df = self.observations()
        df.loc[0, "accuracy"] = np.nan
        with self.assertRaises(Experiment3MetricError):
            global_accuracy_per_round(df)

    def test_weights_and_rank_agreement(self):
        records = [
            {"task": "TinyTask", "seed": 1, "arm": "form_a", "round": 1, "client_id": "c0"},
            {"task": "TinyTask", "seed": 1, "arm": "form_a", "round": 1, "client_id": "c1"},
        ]
        weights = realized_client_weights(records, [0.25, 0.75])
        self.assertAlmostEqual(weights["realized_aggregation_weight"].sum(), 1.0)
        rank_df = weights.assign(
            lambda_weight=[0.5, 1.5],
            intended_domain_divergence_order=[1, 2],
        )
        self.assertAlmostEqual(
            within_round_rank_agreement(rank_df)["lambda_divergence_spearman"].iloc[0],
            1.0,
        )
        one = rank_df.iloc[[0]].copy()
        self.assertEqual(within_round_rank_agreement(one)["lambda_divergence_spearman"].iloc[0], 0.0)


class StatisticsTest(unittest.TestCase):
    def test_paired_differences_and_permutation(self):
        df = pd.DataFrame(
            [
                ["T1", 1, "p1", "baseline", 0.50],
                ["T1", 1, "p1", "form_a", 0.60],
                ["T2", 1, "p2", "baseline", 0.40],
                ["T2", 1, "p2", "form_a", 0.30],
            ],
            columns=["task", "seed", "partition_hash", "arm", "accuracy"],
        )
        paired = paired_arm_differences(
            df,
            baseline_arm="baseline",
            comparison_arm="form_a",
            value_col="accuracy",
        )
        self.assertEqual(list(np.round(paired["difference"], 10)), [0.1, -0.1])
        result = paired_permutation_test(paired["difference"], seed=4)
        self.assertEqual(result.method, "exact")
        self.assertAlmostEqual(result.p_value, 1.0)
        result2 = paired_permutation_test(paired["difference"], seed=4)
        self.assertEqual(result, result2)

    def test_breaking_pairing_is_rejected_or_changes_result(self):
        df = pd.DataFrame(
            [
                ["T1", 1, "p1", "baseline", 0.50],
                ["T1", 1, "p1", "form_a", 0.60],
                ["T1", 1, "p1", "form_a", 0.61],
            ],
            columns=["task", "seed", "partition_hash", "arm", "accuracy"],
        )
        with self.assertRaises(Experiment3StatisticsError):
            paired_arm_differences(df, baseline_arm="baseline", comparison_arm="form_a", value_col="accuracy")

    def test_confidence_interval_mde_and_summaries(self):
        low, high = confidence_interval([1.0, 2.0, 3.0])
        self.assertLess(low, 2.0)
        self.assertGreater(high, 2.0)
        with self.assertRaises(Experiment3StatisticsError):
            compare_to_mde(0.1, mde=None)
        self.assertFalse(compare_to_mde(0.1, mde=0.2)["meets_mde"])
        paired = pd.DataFrame({"task": ["T1", "T1", "T2"], "difference": [0.1, 0.2, -0.1]})
        self.assertEqual(task_level_summary(paired).query("task == 'T1'")["n_pairs"].iloc[0], 2)
        self.assertEqual(pooled_summary(paired)["n_pairs"], 3)


class OutputCheckpointSmokeTest(unittest.TestCase):
    def test_output_directory_confinement_and_validation_before_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_root = root / "repo"
            project_root = repo_root / "project-2-domain-aware-allocation"
            outputs_root = project_root / "outputs"
            exp1_dir = outputs_root / "exp1"
            exp2_dir = outputs_root / "exp2"
            exp3_dir = outputs_root / "exp3"
            exp1_dir.mkdir(parents=True)
            exp2_dir.mkdir()
            exp3_dir.mkdir()
            bad_existing = exp3_dir / "rerun"
            bad_existing.mkdir()
            sentinel = bad_existing / "sentinel.txt"
            sentinel.write_text("stay", encoding="utf-8")
            old_values = (
                exp3_run.PROJECT2_ROOT,
                exp3_run.OUTPUT_ROOT,
                exp3_run.EXP1_DIR,
                exp3_run.EXP2_DIR,
                exp3_run.OUTPUT_DIR,
            )
            try:
                exp3_run.PROJECT2_ROOT = project_root
                exp3_run.OUTPUT_ROOT = outputs_root
                exp3_run.EXP1_DIR = exp1_dir
                exp3_run.EXP2_DIR = exp2_dir
                exp3_run.OUTPUT_DIR = exp3_dir
                with self.assertRaises(CalibrationBundleError):
                    exp3_run.prepare_validated_run(
                        calibration_bundle_path=FIXTURE,
                        output_dir=bad_existing,
                        tasks=["CIFAR-CNN"],
                        experiment1_run_id="exp1-engineering-fixture",
                        experiment2_run_id="exp2-engineering-fixture",
                        run_seed=1,
                        rounds=1,
                        clients=2,
                        overwrite=True,
                    )
            finally:
                (
                    exp3_run.PROJECT2_ROOT,
                    exp3_run.OUTPUT_ROOT,
                    exp3_run.EXP1_DIR,
                    exp3_run.EXP2_DIR,
                    exp3_run.OUTPUT_DIR,
                ) = old_values
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "stay")

            with self.assertRaises(OutputDirectorySafetyError):
                exp3_run.validate_output_scope(PROJECT2_ROOT / "outputs" / "exp2")

    def test_atomic_checkpoint_compatible_and_incompatible_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.json"
            identity = RunIdentity("cfg", "bundle", "commit")
            payload = checkpoint_payload(
                identity=identity,
                task="TinyTask",
                arm="form_a",
                seed=1,
                partition_hash="partition",
                round_id=1,
                client_id="client_0",
                status="round_complete",
                data={"accuracy": 0.5},
            )
            write_checkpoint(path, payload)
            self.assertEqual(read_checkpoint(path, expected_identity=identity)["task"], "TinyTask")
            self.assertFalse(any(child.suffix == ".tmp" for child in path.parent.iterdir()))
            with self.assertRaises(Experiment3CheckpointError):
                read_checkpoint(path, expected_identity=RunIdentity("other", "bundle", "commit"))
            malformed = Path(tmp) / "bad.json"
            malformed.write_text("{", encoding="utf-8")
            with self.assertRaises(Experiment3CheckpointError):
                read_checkpoint(malformed, expected_identity=identity)

    def test_tiny_three_arm_smoke_execution_in_tempdir_only(self):
        before_outputs = set((PROJECT2_ROOT / "outputs").rglob("*"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_root = root / "repo"
            project_root = repo_root / "project-2-domain-aware-allocation"
            outputs_root = project_root / "outputs"
            exp1_dir = outputs_root / "exp1"
            exp2_dir = outputs_root / "exp2"
            exp3_dir = outputs_root / "exp3"
            exp1_dir.mkdir(parents=True)
            exp2_dir.mkdir()
            old_values = (
                exp3_run.PROJECT2_ROOT,
                exp3_run.OUTPUT_ROOT,
                exp3_run.EXP1_DIR,
                exp3_run.EXP2_DIR,
                exp3_run.OUTPUT_DIR,
            )
            ops = InMemoryExperiment3Ops()
            try:
                exp3_run.PROJECT2_ROOT = project_root
                exp3_run.OUTPUT_ROOT = outputs_root
                exp3_run.EXP1_DIR = exp1_dir
                exp3_run.EXP2_DIR = exp2_dir
                exp3_run.OUTPUT_DIR = exp3_dir
                manifest = exp3_run.run_experiment3(
                    Experiment3Config(
                        calibration_bundle_path=FIXTURE,
                        output_dir=exp3_dir,
                        tasks=("TinyTask",),
                        experiment1_run_id="exp1-engineering-fixture",
                        experiment2_run_id="exp2-engineering-fixture",
                        run_seed=3,
                        rounds=1,
                        clients=2,
                        overwrite=False,
                        target_accuracy=50.0,
                        allowed_tasks=("TinyTask",),
                    ),
                    runtime_ops=ops,
                    allow_engineering_fixture=True,
                )
            finally:
                (
                    exp3_run.PROJECT2_ROOT,
                    exp3_run.OUTPUT_ROOT,
                    exp3_run.EXP1_DIR,
                    exp3_run.EXP2_DIR,
                    exp3_run.OUTPUT_DIR,
                ) = old_values

            self.assertEqual(manifest["status"], "complete")
            self.assertFalse(manifest["scientific_results"])
            expected_files = {
                "manifest.json",
                "per_round_client_observations.csv",
                "global_accuracy_per_round.csv",
                "final_round_global_accuracy.csv",
                "per_domain_accuracy.csv",
                "worst_domain_accuracy.csv",
                "fairness_spread_variance.csv",
                "realized_aggregation_weights.csv",
                "lambda_rank_agreement.csv",
                "convergence_speed.csv",
            }
            self.assertTrue(expected_files.issubset({path.name for path in exp3_dir.iterdir()}))
            self.assertEqual(len(list((exp3_dir / "checkpoints").glob("*.json"))), 3)
            baseline_call = [call for call in ops.aggregate_calls if call[0] == "baseline"][0]
            form_a_call = [call for call in ops.aggregate_calls if call[0] == "form_a"][0]
            form_b_call = [call for call in ops.aggregate_calls if call[0] == "form_b"][0]
            self.assertEqual(baseline_call[1], (1.0, 1.0))
            self.assertNotEqual(form_a_call[1], baseline_call[1])
            self.assertNotEqual(form_b_call[1], form_a_call[1])

        after_outputs = set((PROJECT2_ROOT / "outputs").rglob("*"))
        self.assertEqual(before_outputs, after_outputs)

    def test_runner_has_no_public_invalid_calibration_flag(self):
        tree = ast.parse((PROJECT2_ROOT / "experiment" / "experiment3" / "run.py").read_text(encoding="utf-8"))
        flags = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                if node.args and isinstance(node.args[0], ast.Constant):
                    flags.append(str(node.args[0].value))
        self.assertNotIn("--allow-invalid-calibration", flags)
        self.assertNotIn("--allow-synthetic", flags)


if __name__ == "__main__":
    unittest.main()
