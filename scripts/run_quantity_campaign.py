#!/usr/bin/env python3
"""Durable four-GPU scheduler for the fixed gpu003 quantity-skew campaign.

This supervises single-process local-GPU simulations, not a distributed cluster.
It never retries/overwrites failed runs or stops existing processes. A blocked
campaign requires human diagnosis; restarting it does not clear the block.

Example (execute only on gpu003)::

  python scripts/run_quantity_campaign.py --base ~/ahlora-quantity-20260917 \
    --python ~/ahlora-venv/bin/python --adopt /path/to/adopt.json --dry-run

The adoption JSON is a list, or {"jobs": [...]}:
  [{"pid": 130134, "gpu": 0, "arm": "declora16", "seed": 42,
    "partition": "quantity"}, ...]
An inline JSON value is also accepted. On restart, omit --adopt: durable state
and worker receipts recover managed processes. --screen-only completes all eight
seed42 runs and stops before seeds43--46; resume without it to replicate. There
is no accuracy/metric gate. --dry-run writes nothing and starts no subprocesses
except a read-only Python package-version query.

New training/verification workers survive scheduler interruption and write
atomic exit receipts. Already-running, non-child processes have no observable
exit status: completion requires their complete summary and checkpoint files,
followed by the independent verifier. This limitation is explicitly recorded.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time


ARMS = ("declora16", "declora4", "product16_sample", "fixed_uniform",
        "fixed_sample", "adaptive_uniform", "adaptive_sample")
SCREEN_ORDER = ("declora16", "adaptive_sample", "fixed_sample", "declora4",
                "product16_sample", "fixed_uniform", "adaptive_uniform")
GPUS = (0, 1, 2, 3)
RUNTIME_ENV = ("PYTHONPATH", "CUBLAS_WORKSPACE_CONFIG", "CUDA_DEVICE_ORDER",
               "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM",
               "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONHASHSEED")
DEFAULT_CONFIG = dict(task="sst2", rounds=20, local_epochs=1, batch_size=32,
                      max_length=128, lr=0.001, alpha=16.0, max_grad_norm=1.0,
                      max_train=0, threads=2, verify_only=False)


class CampaignError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise CampaignError(message)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    # Match the benchmark's source-manifest aggregate convention.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def absolute(path):
    # Preserve a venv interpreter's symlink path when invoking it.
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def plan(base):
    jobs = []
    for seed in (42, 43, 44, 45, 46):
        for arm in SCREEN_ORDER if seed == 42 else ARMS:
            identifier = f"full-v1-quantity-{arm}-seed{seed}"
            jobs.append(dict(id=identifier, seed=seed, arm=arm, partition="quantity",
                             stage="screen" if seed == 42 else "replication",
                             output=str(base / "runs" / identifier), local_steps=211))
        if seed == 42:
            identifier = "full-v1-equal-declora16-seed42"
            jobs.append(dict(id=identifier, seed=42, arm="declora16", partition="equal",
                             stage="screen", output=str(base / "runs" / identifier), local_steps=0))
    return jobs


def source_files(repo):
    project = repo / "project-3-hierarchical-gossip"
    files = {project / "experiments/quantity_skew_benchmark.py",
             project / "experiments/verify_quantity_checkpoint.py",
             project / "requirements-transformer-gpu.txt"}
    files.update((project / "src").rglob("*.py"))
    files.update({repo / "project-2-domain-aware-allocation/framework/aggregation/domain_weighting.py",
                  repo / "project-2-domain-aware-allocation/experiment/experiment1/signals.py"})
    return {str(path.relative_to(repo)): file_hash(path) for path in sorted(files)}


def package_snapshot(python):
    code = ("import importlib.metadata as m,json,sys; "
            "print(json.dumps({'python':sys.version,'prefix':sys.prefix,"
            "'packages':dict(sorted((d.metadata['Name'].lower().replace('_','-'),d.version) "
            "for d in m.distributions()))},sort_keys=True))")
    result = subprocess.run([str(python), "-c", code], check=True, text=True,
                            capture_output=True, timeout=30)
    return json.loads(result.stdout)


def process_info(pid):
    """Return Linux PID identity; a zombie has finished for scheduling purposes."""
    try:
        folder = Path("/proc") / str(pid)
        stat = (folder / "stat").read_text().rsplit(") ", 1)[1].split()
        if stat[0] in ("Z", "X"):
            return None
        require(folder.stat().st_uid == os.getuid(), f"PID {pid} belongs to another user")
        argv = [arg.decode() for arg in (folder / "cmdline").read_bytes().split(b"\0") if arg]
        if not argv:
            return None
        environment = dict(item.decode().split("=", 1) for item in
                           (folder / "environ").read_bytes().split(b"\0") if b"=" in item)
        return {"pid": pid, "ppid": int(stat[1]), "start_ticks": stat[19], "argv": argv,
                "cwd": str(folder.joinpath("cwd").resolve()),
                "executable": str(folder.joinpath("exe").resolve()),
                "runtime_env": {key: environment.get(key) for key in RUNTIME_ENV},
                "gpu": environment.get("CUDA_VISIBLE_DEVICES")}
    except (FileNotFoundError, ProcessLookupError):
        return None


def option(argv, flag):
    values = [arg.split("=", 1)[1] for arg in argv if arg.startswith(flag + "=")]
    values += [argv[index + 1] for index, arg in enumerate(argv[:-1]) if arg == flag]
    require(len(values) <= 1, f"duplicate process argument {flag}")
    return values[0] if values else None


def process_path(value, info):
    path = Path(value)
    return (Path(info["cwd"]) / path).resolve() if not path.is_absolute() else path.resolve()


def matches_command(info, job, phase, project, python):
    require(info["executable"] == str(python.resolve()), f"PID {info['pid']} uses another interpreter")
    script = project / "experiments" / ("quantity_skew_benchmark.py" if phase == "train"
                                        else "verify_quantity_checkpoint.py")
    require(any(not arg.startswith("-") and process_path(arg, info) == script.resolve()
                for arg in info["argv"][1:]), f"PID {info['pid']} is not the expected {phase} script")
    output = option(info["argv"], "--output" if phase == "train" else "--run")
    require(output is not None and process_path(output, info) == Path(job["output"]).resolve(),
            f"PID {info['pid']} command does not identify {job['output']}")


def parse_adopt(value, jobs):
    if not value:
        return {}
    parsed = json.loads(value) if value.lstrip().startswith(("[", "{")) else read_json(absolute(value))
    parsed = parsed.get("jobs", parsed) if isinstance(parsed, dict) else parsed
    require(isinstance(parsed, list), "adoption JSON must be a list or a jobs list")
    result, pids, gpus = {}, set(), set()
    for item in parsed:
        require(isinstance(item, dict), "each adopted job must be an object")
        matched = [job for job in jobs if (job["arm"], job["seed"], job["partition"]) ==
                   (item.get("arm"), item.get("seed", 42), item.get("partition", "quantity"))]
        require(len(matched) == 1, "adoption does not identify one planned job")
        identifier, pid, gpu = matched[0]["id"], item.get("pid"), item.get("gpu")
        if "output" in item:
            require(absolute(item["output"]) == Path(matched[0]["output"]), "adoption output differs from planned job")
        require(type(pid) is int and pid > 0 and type(gpu) is int and gpu in GPUS, "invalid adopted PID/GPU")
        require(identifier not in result and pid not in pids and gpu not in gpus, "duplicate adoption job, PID or GPU")
        result[identifier] = {"pid": pid, "gpu": gpu}
        pids.add(pid)
        gpus.add(gpu)
    return result


def config_matches(job, base):
    run = Path(job["output"])
    config = read_json(run / "config.json")
    expected = dict(DEFAULT_CONFIG, seed=job["seed"], arm=job["arm"], partition=job["partition"],
                    local_steps=job["local_steps"], assets=str(base / "assets"), output=str(run))
    require(config == expected, f"configuration differs from fixed campaign: {job['id']}")
    return config


def completed_training(job, base, source):
    run = Path(job["output"])
    config_matches(job, base)
    require(read_json(run / "source_manifest.json") == source, f"run source differs: {job['id']}")
    summary = read_json(run / "summary.json")
    require(summary["status"] == "complete" and summary["rounds"] == 20 and
            all(summary[key] == job[key] for key in ("seed", "arm", "partition")),
            f"incomplete or mismatched summary: {job['id']}")
    require(summary["source_sha256"] == object_hash(source), f"summary source differs: {job['id']}")
    for name in ("best", "final"):
        require(file_hash(run / f"{name}.pt") == summary[f"{name}_checkpoint_sha256"],
                f"{name} checkpoint hash mismatch: {job['id']}")
    return summary


def verified(job, base, source, repo):
    summary = completed_training(job, base, source)
    path = Path(job["output"]) / "independent_checkpoint_audit.json"
    if not path.exists():
        return False
    audit = read_json(path)
    require(audit.get("status") == "passed", f"independent audit did not pass: {job['id']}")
    require(Path(audit["run"]).resolve() == Path(job["output"]).resolve(), "audit identifies another run")
    require(audit["evaluator_sha256"] == source["project-3-hierarchical-gossip/experiments/verify_quantity_checkpoint.py"],
            "audit used another verifier")
    require(audit["model_helper_sha256"] == source["project-3-hierarchical-gossip/src/models/roberta_lora.py"],
            "audit used another model helper")
    require(audit["artifact_checks"]["source_sha256"] == object_hash(source) and
            audit["round_metric_records_verified"] == 20, "audit source/round budget mismatch")
    for name in ("best", "final"):
        require(audit["checkpoints"][name]["exact_raw_prediction_match"] is True and
                audit["checkpoints"][name]["checkpoint_sha256"] == summary[f"{name}_checkpoint_sha256"],
                f"audit not bound to current {name} checkpoint")
    return True


def worker(spec_path):
    """Private child entry: durable exit status, even if the scheduler exits."""
    require(socket.gethostname().split(".")[0] == "gpu003", "workers are restricted to gpu003")
    spec_path = absolute(spec_path)
    spec = read_json(spec_path)
    receipt = Path(spec["receipt"])
    require(not receipt.exists(), "worker receipt already exists; refusing duplicate execution")
    environment = os.environ.copy()
    for key, value in spec["runtime_env"].items():
        if value is None:
            environment.pop(key, None)
        else:
            environment[key] = value
    environment["CUDA_VISIBLE_DEVICES"] = str(spec["gpu"])
    record = {"status": "starting", "started_at": now(), "worker_pid": os.getpid(),
              "spec_sha256": file_hash(spec_path), "job_id": spec["job_id"], "phase": spec["phase"]}
    atomic_json(receipt, record)
    try:
        with Path(spec["log"]).open("ab", buffering=0) as log:
            child = subprocess.Popen(spec["command"], cwd=spec["cwd"], env=environment,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     close_fds=True)
            info = process_info(child.pid)
            record.update(status="running", child_pid=child.pid,
                          child_start_ticks=info["start_ticks"] if info else None)
            atomic_json(receipt, record)
            code = child.wait()
        record.update(status="finished", returncode=code, ended_at=now())
        atomic_json(receipt, record)
        return 0 if code == 0 else 1
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}", ended_at=now())
        atomic_json(receipt, record)
        raise


class Campaign:
    def __init__(self, args):
        self.args, self.base, self.python = args, absolute(args.base), absolute(args.python)
        self.repo = self.base / "repo"
        self.project = self.repo / "project-3-hierarchical-gossip"
        self.directory = self.base / "campaign-full-v1"
        self.status_path = self.directory / "status.json"
        self.jobs = plan(self.base)
        self.job_by_id = {job["id"]: job for job in self.jobs}
        self.adopt = parse_adopt(args.adopt, self.jobs)
        self.stop_requested = False
        self.state = None
        self.children = {}
        self.validated_complete = set()
        self.progress_cache = {}
        self.summary_script = absolute(args.summary_script) if args.summary_script else None

    def event(self, kind, **fields):
        item = dict(time=now(), event=kind, **fields)
        with (self.directory / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(item, sort_keys=True), flush=True)

    def save(self):
        self.state["updated_at"] = now()
        durations, remaining = [], 0
        for job in self.jobs:
            path = Path(job["output"]) / "rounds.jsonl"
            if path.exists():
                signature = (path.stat().st_size, path.stat().st_mtime_ns)
                cached = self.progress_cache.get(job["id"])
                if cached is None or cached[0] != signature:
                    rows = []
                    for line in path.read_text().splitlines():
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            # A writer may currently be appending the last line.
                            break
                    cached = (signature, len(rows), [row["round_seconds"] for row in rows])
                    self.progress_cache[job["id"]] = cached
                self.state["jobs"][job["id"]]["rounds_recorded"] = cached[1]
                durations.extend(cached[2])
            remaining += max(0, 20 - self.state["jobs"][job["id"]].get("rounds_recorded", 0))
        self.state["progress"] = {"training_rounds_remaining": remaining,
                                  "verified_jobs": len([r for r in self.state["jobs"].values() if r["status"] == "complete"]),
                                  "planned_jobs": 36,
                                  "estimated_remaining_hours": statistics.median(durations) * remaining / 4 / 3600 if durations else None,
                                  "eta_scope": "Approximate training-only work / four GPUs; excludes verifier/report/startup delays and variable future round cost"}
        atomic_json(self.status_path, self.state)

    def check_pins(self):
        require(source_files(self.repo) == self.state["source_manifest"], "actual benchmark/src/source file set or hashes changed")
        require(file_hash(self.project / "requirements-gpu.txt") == self.state["requirements_gpu_sha256"], "GPU requirements changed")
        require(package_snapshot(self.python) == self.state["python_environment"], "actual Python/package environment changed")
        require(file_hash(self.python.resolve()) == self.state["interpreter_sha256"], "Python executable bytes changed")
        require(file_hash(Path(__file__)) == self.state["scheduler_sha256"], "scheduler source changed during campaign")
        if self.summary_script:
            require(file_hash(self.summary_script) == self.state["summary_script_sha256"], "report script changed during campaign")

    def run_summary(self, reason):
        if self.summary_script is None or self.args.dry_run:
            return
        require(file_hash(self.summary_script) == self.state["summary_script_sha256"], "report script hash mismatch")
        command = [str(self.python), str(self.summary_script), "--inputs", str(self.base / "runs"),
                   "--output", str(self.base / "campaign-summary")]
        log = self.base / "logs" / "campaign-summary.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["MPLBACKEND"] = "Agg"
        self.event("summary_started", reason=reason, command=command, gpu="disabled")
        with log.open("ab", buffering=0) as stream:
            result = subprocess.run(command, cwd=self.repo, env=environment, stdin=subprocess.DEVNULL,
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=600, close_fds=True)
        require(result.returncode == 0, f"CPU summary failed with exit {result.returncode}; see {log}")
        report = self.base / "campaign-summary/summary.json"
        require(report.is_file(), "summary command succeeded without writing summary.json")
        read_json(report)
        self.state["last_summary"] = {"completed_at": now(), "reason": reason, "returncode": 0,
                                      "summary_sha256": file_hash(report), "script_sha256": self.state["summary_script_sha256"]}
        self.save()
        self.event("summary_completed", reason=reason, summary=str(report))

    def initialize(self):
        require(self.python.is_file(), "provided interpreter is missing")
        actual = source_files(self.repo)
        if self.status_path.exists():
            self.state = read_json(self.status_path)
            require(self.state["plan"] == self.jobs and self.state["python_path"] == str(self.python), "campaign plan/interpreter changed")
            require(self.state["status"] != "blocked", "campaign is blocked; preserve it and diagnose before any manual recovery")
            if self.summary_script is None and self.state.get("summary_script"):
                self.summary_script = Path(self.state["summary_script"])
            require((str(self.summary_script) if self.summary_script else None) == self.state.get("summary_script"),
                    "summary script path changed; cannot change reporting silently on resume")
            self.check_pins()
            return
        # Bind the campaign to an actual initial launched run, not just whatever
        # source happens to be present when the scheduler first starts.
        first = self.base / "runs/full-v1-quantity-declora16-seed42/source_manifest.json"
        require(first.is_file(), "initial seed42 Dec-LoRA source manifest is required")
        require(read_json(first) == actual, "actual source differs from initial launched-run manifest")
        runtime = {key: None for key in RUNTIME_ENV}
        runtime.update(PYTHONPATH=f"{Path.home() / 'pystubs'}:{self.project}", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        for adopted in self.adopt.values():
            info = process_info(adopted["pid"])
            if info:
                runtime = info["runtime_env"]
                break
        require(runtime["PYTHONPATH"] == f"{Path.home() / 'pystubs'}:{self.project}", "unexpected adopted PYTHONPATH")
        self.state = dict(schema_version=1, status="initializing", created_at=now(), updated_at=now(),
                          host=socket.gethostname(), base=str(self.base), python_path=str(self.python),
                          scheduler_sha256=file_hash(Path(__file__)), interpreter_sha256=file_hash(self.python.resolve()),
                          source_manifest=actual, source_sha256=object_hash(actual),
                          requirements_gpu_sha256=file_hash(self.project / "requirements-gpu.txt"),
                          summary_script=str(self.summary_script) if self.summary_script else None,
                          summary_script_sha256=file_hash(self.summary_script) if self.summary_script else None,
                          python_environment=package_snapshot(self.python), runtime_env=runtime, plan=self.jobs,
                          jobs={job["id"]: {"status": "pending"} for job in self.jobs},
                          scope="Four local GPU workers; single-process peer simulations; no metric gate")

    def validate_external(self, info, job, gpu, phase):
        matches_command(info, job, phase, self.project, self.python)
        require(info["gpu"] == str(gpu), f"PID {info['pid']} does not use assigned GPU {gpu}")
        require(info["runtime_env"] == self.state["runtime_env"], f"PID {info['pid']} runtime environment differs")

    def discover(self, spec_path):
        found = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if entry.stat().st_uid != os.getuid():
                    continue
                info = process_info(int(entry.name))
                if info and option(info["argv"], "--_worker-spec") == str(spec_path):
                    found.append(info)
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
        require(len(found) <= 1, f"duplicate worker for {spec_path}")
        return found[0] if found else None

    def active_info(self, record):
        pid = record.get("pid")
        info = process_info(pid) if pid else None
        if info and record.get("start_ticks"):
            require(info["start_ticks"] == record["start_ticks"], "PID was reused; refusing to monitor another process")
        return info

    def reconcile(self):
        used = set()
        for job in self.jobs:
            identifier, run = job["id"], Path(job["output"])
            record = self.state["jobs"][identifier]
            adoption = self.adopt.get(identifier)
            if record["status"] == "complete" and identifier in self.validated_complete:
                continue
            if record["status"] == "failed":
                raise CampaignError(f"recorded failure for {identifier}")
            if record.get("spec"):
                spec_path = Path(record["spec"])
                require(file_hash(spec_path) == record["spec_sha256"], "worker specification changed")
                if record["status"].startswith("launching_") and not record.get("pid"):
                    info = self.discover(spec_path)
                    if info:
                        record.update(pid=info["pid"], start_ticks=info["start_ticks"])
            info = self.active_info(record)
            if adoption and record["status"] in ("pending", "awaiting_verification"):
                info = process_info(adoption["pid"])
                if info:
                    self.validate_external(info, job, adoption["gpu"], "train")
                    config_matches(job, self.base)
                    require(read_json(run / "source_manifest.json") == self.state["source_manifest"], "adopted source mismatch")
                    record.update(status="training", phase="train", kind="adopted", gpu=adoption["gpu"],
                                  pid=info["pid"], start_ticks=info["start_ticks"], adopted_at=now(),
                                  exit_code_scope="non-child exit code unavailable; require complete artifacts and standalone audit")
                elif not (run / "summary.json").exists():
                    raise CampaignError(f"adopted PID exited without completed summary: {identifier}")
            if info:
                require(record.get("gpu") in GPUS and record["gpu"] not in used, "more than one managed worker on a GPU")
                if record.get("kind") == "adopted":
                    self.validate_external(info, job, record["gpu"], record["phase"])
                else:
                    require(option(info["argv"], "--_worker-spec") == record.get("spec"), "live worker commandline/spec mismatch")
                    require(info["executable"] == str(self.python.resolve()), "worker interpreter mismatch")
                used.add(record["gpu"])
                continue
            if record.get("kind") == "worker" and record["status"] != "complete":
                receipt_path = Path(record["receipt"])
                require(receipt_path.is_file(), f"worker disappeared without receipt: {identifier}")
                receipt = read_json(receipt_path)
                require(receipt["spec_sha256"] == record["spec_sha256"], "receipt/spec hash mismatch")
                require(receipt["job_id"] == identifier and receipt["phase"] == record["phase"], "receipt phase/job mismatch")
                if receipt["status"] != "finished" or receipt.get("returncode") != 0:
                    record["status"] = "failed"
                require(receipt["status"] == "finished" and receipt.get("returncode") == 0,
                        f"worker runtime failure/incomplete exit for {identifier}: {receipt}")
                record["last_returncode"] = 0
            if (run / "summary.json").exists():
                completed = verified(job, self.base, self.state["source_manifest"], self.repo)
                if completed:
                    newly_complete = record["status"] != "complete"
                    if newly_complete and not self.args.dry_run:
                        self.event("verified_complete", job=identifier, gpu=record.get("gpu"))
                    record.update(status="complete", completed_at=record.get("completed_at", now()))
                    self.validated_complete.add(identifier)
                    if newly_complete:
                        self.run_summary("verified:" + identifier)
                elif record.get("phase") == "verify":
                    raise CampaignError(f"verifier exited without a passing audit: {identifier}")
                else:
                    record["status"] = "awaiting_verification"
                record.pop("pid", None)
                record.pop("start_ticks", None)
            elif run.exists():
                raise CampaignError(f"incomplete run has no live adopted/managed process: {identifier}")
            elif record["status"] != "pending":
                raise CampaignError(f"launched job produced no completed data: {identifier}")
        self.check_unmanaged()
        return used

    def check_unmanaged(self):
        """Do not start around an unadopted job that targets a planned output."""
        outputs = {job["output"]: job for job in self.jobs}
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if entry.stat().st_uid != os.getuid():
                    continue
                info = process_info(int(entry.name))
                if not info:
                    continue
                if not any(Path(arg).name in ("quantity_skew_benchmark.py", "verify_quantity_checkpoint.py")
                           for arg in info["argv"][1:] if not arg.startswith("-")):
                    continue
                value = option(info["argv"], "--output") or option(info["argv"], "--run")
                if value is None:
                    continue
                output = str(process_path(value, info))
                if output not in outputs:
                    continue
                record = self.state["jobs"][outputs[output]["id"]]
                managed = info["pid"] == record.get("pid") and record.get("kind") == "adopted"
                if record.get("kind") == "worker":
                    parent = self.active_info(record)
                    if parent and info["ppid"] == parent["pid"]:
                        # A newly spawned child can appear in /proc before its
                        # worker atomically records child_pid. Authorize only a
                        # direct child of the exact pinned, managed worker.
                        require(parent["executable"] == str(self.python.resolve()), "worker interpreter mismatch")
                        require(option(parent["argv"], "--_worker-spec") == record.get("spec"),
                                "child parent is not the managed worker")
                        require(file_hash(record["spec"]) == record["spec_sha256"], "worker specification changed")
                        managed = True
                    elif record.get("receipt") and Path(record["receipt"]).is_file():
                        receipt = read_json(record["receipt"])
                        require(receipt["spec_sha256"] == record["spec_sha256"] and
                                receipt["job_id"] == outputs[output]["id"] and
                                receipt["phase"] == record["phase"], "child receipt identity mismatch")
                        managed = (info["pid"] == receipt.get("child_pid") and
                                   info["start_ticks"] == receipt.get("child_start_ticks"))
                require(managed, f"unadopted live process {info['pid']} targets {output}")
                self.validate_external(info, outputs[output], record["gpu"], record["phase"])
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue

    def launch(self, job, gpu, phase):
        self.check_pins()
        self.check_unmanaged()
        record = self.state["jobs"][job["id"]]
        run = Path(job["output"])
        if phase == "train":
            require(not run.exists(), "refusing to overwrite an existing run")
            command = [str(self.python), "-u", str(self.project / "experiments/quantity_skew_benchmark.py"),
                       "--assets", str(self.base / "assets"), "--output", str(run),
                       "--arm", job["arm"], "--seed", str(job["seed"]), "--partition", job["partition"],
                       "--rounds", "20", "--local-epochs", "1", "--local-steps", str(job["local_steps"])]
        else:
            completed_training(job, self.base, self.state["source_manifest"])
            command = [str(self.python), "-u", str(self.project / "experiments/verify_quantity_checkpoint.py"),
                       "--run", str(run), "--assets", str(self.base / "assets")]
        stem = self.directory / "workers" / job["id"] / phase
        spec_path, receipt = stem.with_suffix(".spec.json"), stem.with_suffix(".receipt.json")
        require(not spec_path.exists() and not receipt.exists(), "phase has already been launched; no silent retry")
        log = self.base / "logs" / f"{job['id']}-{phase}-managed.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        spec = dict(job_id=job["id"], output=str(run), phase=phase, gpu=gpu, command=command,
                    cwd=str(self.project), runtime_env=self.state["runtime_env"], receipt=str(receipt),
                    log=str(log), source_sha256=self.state["source_sha256"], created_at=now())
        atomic_json(spec_path, spec)
        record.update(status="launching_" + phase, phase=phase, kind="worker", gpu=gpu,
                      spec=str(spec_path), spec_sha256=file_hash(spec_path), receipt=str(receipt), log=str(log))
        record.pop("pid", None)
        record.pop("start_ticks", None)
        self.save()
        self.event("launch_intent", job=job["id"], phase=phase, gpu=gpu, command=command)
        with log.open("ab", buffering=0) as stream:
            child = subprocess.Popen([str(self.python), str(Path(__file__).resolve()), "--_worker-spec", str(spec_path)],
                                     cwd=str(self.project), stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)
        self.children[child.pid] = child
        info = process_info(child.pid)
        record.update(pid=child.pid, start_ticks=info["start_ticks"] if info else None,
                      status="training" if phase == "train" else "verifying", launched_at=now())
        self.save()
        self.event("launched", job=job["id"], phase=phase, gpu=gpu, pid=child.pid)

    def execute(self):
        self.initialize()
        used = self.reconcile()
        if self.args.dry_run:
            print(json.dumps({"status": "dry_run_passed", "writes": False, "launches": False,
                              "source_sha256": self.state["source_sha256"], "occupied_gpus": sorted(used),
                              "screen_only": self.args.screen_only,
                              "jobs": [{**job, "state": self.state["jobs"][job["id"]]} for job in self.jobs]}, indent=2))
            return 0
        self.state.update(status="running", scheduler_pid=os.getpid(), screen_only=self.args.screen_only)
        self.save()
        self.event("scheduler_started", pid=os.getpid(), screen_only=self.args.screen_only)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: setattr(self, "stop_requested", True))
        while not self.stop_requested:
            for pid, child in list(self.children.items()):
                if child.poll() is not None:
                    del self.children[pid]
            used = self.reconcile()
            self.save()
            screen_done = all(self.state["jobs"][job["id"]]["status"] == "complete"
                              for job in self.jobs if job["stage"] == "screen")
            if all(record["status"] == "complete" for record in self.state["jobs"].values()):
                self.run_summary("campaign_complete")
                self.state["status"] = "complete"
                self.save()
                self.event("campaign_complete", verified_jobs=36)
                return 0
            if screen_done and self.args.screen_only:
                self.run_summary("screen_complete")
                self.state["status"] = "screen_complete"
                self.save()
                self.event("screen_complete", verified_jobs=8, replication="not launched (--screen-only)")
                return 0
            # Verification retains the training GPU when that assignment is known.
            for job in self.jobs:
                record = self.state["jobs"][job["id"]]
                if record["status"] != "awaiting_verification":
                    continue
                available = [gpu for gpu in GPUS if gpu not in used]
                gpu = record.get("gpu", available[0] if available else None)
                if gpu in available:
                    self.launch(job, gpu, "verify")
                    used.add(gpu)
            for gpu in GPUS:
                if gpu in used:
                    continue
                candidate = next((job for job in self.jobs
                                  if self.state["jobs"][job["id"]]["status"] == "pending"
                                  and (job["stage"] == "screen" or screen_done)), None)
                if candidate:
                    self.launch(candidate, gpu, "train")
                    used.add(gpu)
            # Short sleeps keep signal handling responsive. Workers are detached;
            # stopping the scheduler never sends them a signal.
            deadline = time.monotonic() + self.args.poll_seconds
            while not self.stop_requested and time.monotonic() < deadline:
                time.sleep(min(1, max(0, deadline - time.monotonic())))
        self.state["status"] = "paused"
        self.save()
        self.event("scheduler_paused", workers="left running; resume with the same arguments")
        return 0


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--_worker-spec":
        return worker(sys.argv[2])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adopt", help="Path to adoption JSON, or inline JSON; omit when resuming durable state")
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--summary-script", type=Path,
                        help="Pinned CPU report script; run after each verified completion and at stage/campaign end")
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()
    require(socket.gethostname().split(".")[0] == "gpu003", "Scheduler execution is restricted to gpu003")
    require(1 <= args.poll_seconds <= 60, "poll interval must be between 1 and 60 seconds")
    campaign = Campaign(args)
    # Lock the existing base directory itself, closing first-launch and dry-run
    # races without creating a lock file in read-only dry-run mode.
    require(campaign.base.is_dir(), "campaign base must already exist")
    directory_fd = os.open(campaign.base, os.O_RDONLY)
    try:
        try:
            fcntl.flock(directory_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise CampaignError("another scheduler holds this campaign base lock")
        if not args.dry_run:
            campaign.directory.mkdir(parents=True, exist_ok=True)
        try:
            return campaign.execute()
        except Exception as error:
            if not args.dry_run and campaign.state is not None:
                campaign.state.update(status="blocked", error=f"{type(error).__name__}: {error}", blocked_at=now())
                campaign.save()
                campaign.event("campaign_blocked", error=campaign.state["error"], workers="not stopped")
            raise
    finally:
        os.close(directory_fd)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CampaignError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"quantity campaign stopped: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
