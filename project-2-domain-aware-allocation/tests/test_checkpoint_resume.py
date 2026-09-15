import importlib.util
import json
from pathlib import Path
import pytest
from types import SimpleNamespace

pytest.importorskip("torch")


RUN_PATH = Path(__file__).parents[1] / "experiment" / "experiment1" / "run.py"
spec = importlib.util.spec_from_file_location("p2_exp1_run", RUN_PATH)
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def _args(rounds=5, seed=42):
    return SimpleNamespace(
        num_rounds=rounds,
        seed=seed,
        data_root=Path("/tmp/data"),
        synthetic_datasets=[],
        download_datasets=False,
    )


def _partition():
    return run.PartitionConfig(strategy="iid", alpha=0.5, seed=42, num_workers=0, pin_memory=False)


def _manifest(tasks=("CIFAR-CNN",)):
    return {"datasets": {task: {"synthetic": False, "dataset_id": task} for task in tasks}}


def _checkpoint_payload(tasks=("CIFAR-CNN",), rounds=5, seed=42):
    args = _args(rounds=rounds, seed=seed)
    task_names = list(tasks)
    identity = run._run_identity(task_names, args, _partition(), _manifest(task_names))
    return {
        "schema_version": run.CHECKPOINT_SCHEMA_VERSION,
        "output_schema_version": run.CHECKPOINT_OUTPUT_SCHEMA_VERSION,
        "resume_boundary": run.CHECKPOINT_RESUME_BOUNDARY,
        "status": "running",
        "tasks": task_names,
        "num_rounds": rounds,
        "num_clients": run.NUM_CLIENTS,
        "seed": seed,
        "partition": run._json_safe(_partition().__dict__),
        "data_root": str(Path("/tmp/data").resolve()),
        "synthetic_datasets": [],
        "download_datasets": False,
        "task_seed_policy": "seed + task index",
        "run_identity": identity,
        "run_identity_hash": run._stable_hash(identity),
        "dataset_provenance": identity["dataset_provenance"],
        "source_revision": identity["source_revision"],
        "completed_tasks": [],
        "task_results": [],
    }


def test_atomic_checkpoint_write_and_load(tmp_path):
    path = tmp_path / "checkpoint.json"
    payload = _checkpoint_payload()
    run._atomic_json_write(path, payload)
    assert json.loads(path.read_text()) == payload
    assert not path.with_name("checkpoint.json.tmp").exists()
    assert run._load_checkpoint(path) == payload


def test_checkpoint_identity_rejects_configuration_mismatch():
    checkpoint = _checkpoint_payload()
    run._validate_checkpoint_identity(
        checkpoint,
        task_names=["CIFAR-CNN"],
        args=_args(),
        partition_config=_partition(),
        dataset_manifest=_manifest(),
    )
    bad = dict(checkpoint, seed=7)
    try:
        run._validate_checkpoint_identity(bad, task_names=["CIFAR-CNN"], args=_args(), partition_config=_partition())
    except ValueError as exc:
        assert "seed" in str(exc)
    else:
        raise AssertionError("mismatched seed was accepted")


def test_checkpoint_loader_rejects_malformed(tmp_path):
    path = tmp_path / "checkpoint.json"
    path.write_text("not json")
    try:
        run._load_checkpoint(path)
    except ValueError:
        pass
    else:
        raise AssertionError("malformed checkpoint was accepted")


def test_checkpoint_loader_rejects_incomplete(tmp_path):
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps({"schema_version": run.CHECKPOINT_SCHEMA_VERSION, "status": "running"}))
    try:
        run._load_checkpoint(path)
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete checkpoint was accepted")


def test_checkpoint_loader_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "checkpoint.json"
    path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(run.Experiment1CheckpointError, match="duplicate JSON key"):
        run._load_checkpoint(path)


def test_checkpoint_loader_rejects_nested_duplicate_keys(tmp_path):
    path = tmp_path / "checkpoint.json"
    path.write_text('{"schema_version":1,"status":"running","run_identity":{"seed":1,"seed":2}}', encoding="utf-8")
    with pytest.raises(run.Experiment1CheckpointError, match="duplicate JSON key"):
        run._load_checkpoint(path)


def test_checkpoint_loader_rejects_tampered_run_identity(tmp_path):
    path = tmp_path / "checkpoint.json"
    payload = _checkpoint_payload()
    payload["run_identity"]["seed"] = 7
    run._atomic_json_write(path, payload)
    with pytest.raises(ValueError, match="hash"):
        run._load_checkpoint(path)


def test_resume_skips_completed_task_after_interruption(tmp_path, monkeypatch):
    tasks = ["CIFAR-CNN", "Fashion-MLP"]
    args = _args(rounds=1)
    args.output_dir = tmp_path / "run"
    args.resume = False
    args.resume_from = None
    args.overwrite = False
    args.partition = "iid"
    args.alpha = 0.5
    args.num_workers = 0
    args.pin_memory = False
    args.tasks = tasks
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run, "parse_args", lambda: args)
    monkeypatch.setattr(run, "_source_revision", lambda: {"git_commit": "test", "runner_sha256": "test"})
    monkeypatch.setattr(run, "load_experiments", lambda **kwargs: ([(task, None, None, None) for task in tasks], {}))
    monkeypatch.setattr(run, "write_dataset_manifest", lambda **kwargs: {"datasets": {task: {"synthetic": True} for task in tasks}})
    monkeypatch.setattr(run, "save_label_distribution_outputs", lambda *args: run.pd.DataFrame())
    monkeypatch.setattr(run, "run_statistical_analysis", lambda *args: ([], []))
    monkeypatch.setattr(run, "plot_signal_vs_contribution", lambda *args: None)

    def result(task):
        rows = [dict.fromkeys(run.MEASUREMENT_NUMERIC_FIELDS, 1.0) for _ in range(run.NUM_CLIENTS)]
        for client_id, row in enumerate(rows):
            row.update(task=task, is_synthetic=True, partition_strategy="iid", round=1, client_id=client_id)
        labels = [{"task": task, "client_id": client_id} for client_id in range(run.NUM_CLIENTS)]
        payload = {"task": task, "global_class_counts": [1], "global_class_frequency": [1.0],
                   "client_class_counts": [[1]] * run.NUM_CLIENTS, "client_class_frequency": [[1.0]] * run.NUM_CLIENTS}
        return {"task": task, "is_synthetic": True, "label_records": labels, "label_payload": payload, "round_rows": rows, "accuracy_curve": [50.0]}

    calls = []

    def interrupted_task(**kwargs):
        task = kwargs["task_name"]
        calls.append(task)
        if task == tasks[1]:
            raise RuntimeError("simulated interruption")
        return result(task)

    monkeypatch.setattr(run, "run_task", interrupted_task)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run.main()
    checkpoint = run._load_checkpoint(args.output_dir / "checkpoint.json")
    assert checkpoint["completed_tasks"] == tasks[:1]

    calls.clear()
    args.resume = True

    def resumed_task(**kwargs):
        calls.append(kwargs["task_name"])
        return result(kwargs["task_name"])

    monkeypatch.setattr(run, "run_task", resumed_task)
    run.main()
    assert calls == tasks[1:]
    checkpoint = run._load_checkpoint(args.output_dir / "checkpoint.json")
    assert checkpoint["status"] == "completed"
    assert checkpoint["completed_tasks"] == tasks


def test_final_checkpoint_precedes_run_completed_progress_failure(tmp_path, monkeypatch):
    tasks = ["CIFAR-CNN"]
    args = _args(rounds=1)
    args.output_dir = tmp_path / "run"
    args.resume = False
    args.resume_from = None
    args.overwrite = False
    args.partition = "iid"
    args.alpha = 0.5
    args.num_workers = 0
    args.pin_memory = False
    args.tasks = tasks
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run, "parse_args", lambda: args)
    monkeypatch.setattr(run, "_source_revision", lambda: {"git_commit": "test", "runner_sha256": "test"})
    monkeypatch.setattr(run, "load_experiments", lambda **kwargs: ([(task, None, None, None) for task in tasks], {}))
    monkeypatch.setattr(run, "write_dataset_manifest", lambda **kwargs: {"datasets": {task: {"synthetic": True} for task in tasks}})
    monkeypatch.setattr(run, "save_label_distribution_outputs", lambda *args: run.pd.DataFrame())
    monkeypatch.setattr(run, "run_statistical_analysis", lambda *args: ([], []))
    monkeypatch.setattr(run, "plot_signal_vs_contribution", lambda *args: None)

    def result(task):
        rows = [dict.fromkeys(run.MEASUREMENT_NUMERIC_FIELDS, 1.0) for _ in range(run.NUM_CLIENTS)]
        for client_id, row in enumerate(rows):
            row.update(task=task, is_synthetic=True, partition_strategy="iid", round=1, client_id=client_id)
        labels = [{"task": task, "client_id": client_id} for client_id in range(run.NUM_CLIENTS)]
        payload = {"task": task, "global_class_counts": [1], "global_class_frequency": [1.0],
                   "client_class_counts": [[1]] * run.NUM_CLIENTS, "client_class_frequency": [[1.0]] * run.NUM_CLIENTS}
        return {"task": task, "is_synthetic": True, "label_records": labels, "label_payload": payload, "round_rows": rows, "accuracy_curve": [50.0]}

    monkeypatch.setattr(run, "run_task", lambda **kwargs: result(kwargs["task_name"]))

    original_record_progress = run._record_progress

    def fail_on_completed(output_dir, event, **payload):
        if event == "run_completed":
            raise RuntimeError("simulated progress append crash")
        return original_record_progress(output_dir, event, **payload)

    monkeypatch.setattr(run, "_record_progress", fail_on_completed)
    with pytest.raises(RuntimeError, match="progress append"):
        run.main()

    checkpoint = run._load_checkpoint(args.output_dir / "checkpoint.json")
    assert checkpoint["status"] == "completed"
    assert checkpoint["completed_tasks"] == tasks
