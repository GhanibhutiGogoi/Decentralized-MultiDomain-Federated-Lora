"""Run the fixed completion battery and create its evidence report.

Execute on the documented SSH GPU machine, from any working directory:
    python experiments/run_completion.py --output-root results/completion \
        --data-dir data --feature-cache data/features --device cuda

The output root must be empty. Jobs run serially, stop at the first failure, and
never resume or overwrite existing evidence. Inspect battery_status.json and
logs/*.log for progress. The same verified feature cache is reused by every job.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTED_ENVIRONMENT = ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def build_jobs(output_root, data_dir, feature_cache, device, python=None):
    """Return the exact argument lists; this function never runs training."""
    python = python or sys.executable
    output_root, data_dir, feature_cache = [Path(path).expanduser().resolve()
                                          for path in (output_root, data_dir, feature_cache)]
    common = ["--data-dir", str(data_dir), "--feature-cache", str(feature_cache),
        "--device", device, "--seeds", "42", "43", "44", "--alpha", "32", "--consensus-rank", "16",
        "--n-domains", "5", "--clients-per-domain", "3", "--dirichlet-alpha", "0.5",
        "--bridge-every", "5", "--topology", "ring", "--local-epochs", "1", "--lr", "0.001",
        "--weight-decay", "0.0001", "--batch-size", "128", "--eval-batch-size", "2048",
        "--image-size", "224", "--feature-batch-size", "256", "--num-workers", "2",
        "--torch-threads", "4", "--max-train-per-client", "0", "--max-test-per-domain", "0"]
    jobs = []
    def job(name, script, extra, kind="experiment"):
        destination = output_root / name
        jobs.append({"name": name, "kind": kind, "status": "pending",
                     "output": str(destination), "cwd": str(PROJECT_ROOT),
                     "argv": [python, "-u", str(PROJECT_ROOT / "experiments" / script),
                              "--output", str(destination), *extra],
                     "log": str(output_root / "logs" / f"{len(jobs) + 1:02d}_{name}.log"),
                     "completion_record": str(destination / ("aggregate.json" if kind == "report" else "manifest.json"))})
    for name, ranks, methods, merge, feedback in (
        ("uniform", ["16"], ["local", "fedavg", "mh", "oracle"], "delta", False),
        ("heterogeneous_delta", ["4", "12", "32"], ["local", "fedavg", "mh", "oracle"], "delta", False),
        ("heterogeneous_factor", ["4", "12", "32"], ["fedavg", "mh", "oracle"], "factor_zero_pad", False),
        ("heterogeneous_feedback", ["4", "12", "32"], ["mh", "oracle"], "delta", True),
    ):
        arguments = [*common, "--rounds", "50", "--ranks", *ranks, "--methods", *methods, "--merge", merge]
        if feedback:
            arguments.append("--error-feedback")
        job(name, "04_protocol_benchmark.py", arguments)
    job("report", "summarize_completion.py", ["--inputs", *[item["output"] for item in jobs]], kind="report")
    job("signatures", "05_signature_validation.py", [*common, "--ranks", "16", "--methods", "local", "mh",
        "--merge", "delta", "--stages", "2", "5", "10", "20", "--n-clusters", "5"])
    return jobs


def execute_jobs(status, status_path, environment, child_runner=None):
    """Persist every transition and require a successful child's completion marker."""
    run = child_runner or subprocess.run
    status["status"] = "running"
    status["started_unix"] = time.time()
    write_json(status_path, status)
    for job in status["jobs"]:
        started = time.perf_counter()
        job["status"], job["started_unix"] = "running", time.time()
        status["active_job"] = job["name"]
        write_json(status_path, status)
        print(f"Starting {job['name']}; log: {job['log']}", flush=True)
        try:
            with Path(job["log"]).open("x") as log:
                completed = run(job["argv"], cwd=job["cwd"], env=environment,
                                stdout=log, stderr=subprocess.STDOUT, check=False)
            job["returncode"] = completed.returncode
            if completed.returncode != 0:
                raise subprocess.CalledProcessError(completed.returncode, job["argv"])
            marker = Path(job["completion_record"])
            if not marker.exists():
                raise RuntimeError(f"{job['name']} exited successfully without its completion record: {marker}")
            if json.loads(marker.read_text()).get("status") != "complete":
                raise RuntimeError(f"{job['name']} exited successfully but its completion record is not complete: {marker}")
            job["status"] = "complete"
            print(f"Completed {job['name']}", flush=True)
        except (Exception, KeyboardInterrupt) as error:
            job["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            job["error"] = {"type": type(error).__name__, "message": str(error)}
            status["status"] = job["status"]
            status["failed_job"] = job["name"]
            status["active_job"] = None
            status["finished_unix"] = time.time()
            raise
        finally:
            job["finished_unix"] = time.time()
            job["wall_seconds"] = time.perf_counter() - started
            status["wall_seconds"] = time.time() - status["started_unix"]
            write_json(status_path, status)
    status["status"], status["active_job"] = "complete", None
    status["finished_unix"] = time.time()
    status["wall_seconds"] = status["finished_unix"] - status["started_unix"]
    write_json(status_path, status)
    return status


def main(argv=None, child_runner=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--feature-cache", type=Path, default=Path("data/features"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"output root must be empty; existing evidence is never overwritten: {output}")
    (output / "logs").mkdir()
    environment = os.environ.copy()
    # Respect explicit operator choices; bound BLAS oversubscription otherwise.
    for key, value in {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1"}.items():
        environment.setdefault(key, value)
    jobs = build_jobs(output, args.data_dir, args.feature_cache, args.device)
    status = {"schema_version": 1, "status": "planned", "created_unix": time.time(),
        "output_root": str(output), "invocation": {"executable": sys.executable,
            "argv": sys.argv if argv is None else [sys.argv[0], *argv], "cwd": str(Path.cwd())},
        "orchestrator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {key: environment.get(key) for key in SELECTED_ENVIRONMENT},
        "protocol": {"seeds": [42, 43, 44], "benchmark_rounds": 50, "alpha": 32,
            "consensus_rank": 16, "uniform_rank": 16, "heterogeneous_ranks": [4, 12, 32],
            "signature_stages": [2, 5, 10, 20], "signature_rank": 16,
            "training": "fixed frozen-feature protocol, all official CIFAR-100 data; exact child arguments recorded",
            "execution": "serial, stop on first failure, no resume or overwrite",
            "report": "aggregate all completed Experiment04 arms before running offline signature validation"},
        "jobs": jobs}
    status_path = output / "battery_status.json"
    write_json(status_path, status)
    return execute_jobs(status, status_path, environment, child_runner)


if __name__ == "__main__":
    main()
