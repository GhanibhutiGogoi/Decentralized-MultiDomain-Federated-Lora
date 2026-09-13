"""Validate orchestration without starting any training or subprocess."""

import json
from pathlib import Path
import subprocess

import pytest

from experiments import run_completion as completion


def values(argv, option):
    start = argv.index(option) + 1
    end = next((index for index in range(start, len(argv)) if argv[index].startswith("--")), len(argv))
    return argv[start:end]


def success(argv, **kwargs):
    destination = Path(values(argv, "--output")[0])
    destination.mkdir()
    marker = "aggregate.json" if destination.name == "report" else "manifest.json"
    (destination / marker).write_text(json.dumps({"status": "complete"}))
    kwargs["stdout"].write("mocked child completed; no training ran\n")
    return subprocess.CompletedProcess(argv, 0)


def test_fixed_plan_covers_actual_battery_and_report_inputs(tmp_path):
    jobs = completion.build_jobs(tmp_path / "output", tmp_path / "data", tmp_path / "cache", "cuda", python="python")
    assert [job["name"] for job in jobs] == ["uniform", "heterogeneous_delta", "heterogeneous_factor",
        "heterogeneous_feedback", "report", "signatures"]
    for job in jobs[:4]:
        assert values(job["argv"], "--seeds") == ["42", "43", "44"]
        assert values(job["argv"], "--rounds") == ["50"]
        assert values(job["argv"], "--alpha") == ["32"]
        assert values(job["argv"], "--consensus-rank") == ["16"]
    assert values(jobs[0]["argv"], "--ranks") == ["16"]
    assert values(jobs[1]["argv"], "--ranks") == ["4", "12", "32"]
    assert values(jobs[1]["argv"], "--methods") == ["local", "fedavg", "mh", "oracle"]
    assert values(jobs[2]["argv"], "--merge") == ["factor_zero_pad"]
    assert values(jobs[2]["argv"], "--methods") == ["fedavg", "mh", "oracle"]
    assert "--error-feedback" in jobs[3]["argv"]
    assert values(jobs[4]["argv"], "--inputs") == [job["output"] for job in jobs[:4]]
    assert values(jobs[5]["argv"], "--stages") == ["2", "5", "10", "20"]
    assert "--rounds" not in jobs[5]["argv"]
    assert len({values(job["argv"], "--feature-cache")[0] for job in jobs if job["kind"] == "experiment"}) == 1


def test_success_records_all_commands_logs_and_final_status(tmp_path):
    output = tmp_path / "battery"
    status = completion.main(["--output-root", str(output), "--device", "cpu"], child_runner=success)
    persisted = json.loads((output / "battery_status.json").read_text())
    assert status == persisted
    assert persisted["status"] == "complete"
    assert persisted["active_job"] is None
    assert all(job["status"] == "complete" and job["returncode"] == 0 for job in persisted["jobs"])
    assert all(Path(job["log"]).read_text().startswith("mocked child") for job in persisted["jobs"])
    assert not (output / "battery_status.json.tmp").exists()
    with pytest.raises(FileExistsError, match="never overwritten"):
        completion.main(["--output-root", str(output)], child_runner=success)


def test_failure_preserves_completed_job_and_stops_remaining_jobs(tmp_path):
    calls = []
    def fail_second(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 2:
            kwargs["stdout"].write("specific child failure\n")
            return subprocess.CompletedProcess(argv, 7)
        return success(argv, **kwargs)
    output = tmp_path / "failed"
    with pytest.raises(subprocess.CalledProcessError) as error:
        completion.main(["--output-root", str(output)], child_runner=fail_second)
    assert error.value.returncode == 7
    status = json.loads((output / "battery_status.json").read_text())
    assert len(calls) == 2
    assert status["status"] == "failed" and status["failed_job"] == "heterogeneous_delta"
    assert [job["status"] for job in status["jobs"]] == ["complete", "failed", "pending", "pending", "pending", "pending"]
    assert "specific child failure" in Path(status["jobs"][1]["log"]).read_text()


def test_zero_exit_without_completion_marker_is_not_success(tmp_path):
    output = tmp_path / "missing-marker"
    with pytest.raises(RuntimeError, match="without its completion record"):
        completion.main(["--output-root", str(output)],
            child_runner=lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0))
    status = json.loads((output / "battery_status.json").read_text())
    assert status["status"] == "failed"
    assert status["jobs"][0]["returncode"] == 0
    assert status["jobs"][0]["error"]["type"] == "RuntimeError"
