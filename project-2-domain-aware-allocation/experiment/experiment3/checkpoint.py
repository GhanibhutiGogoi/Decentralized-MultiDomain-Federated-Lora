"""Atomic checkpoint and resume contract for Experiment 3."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


CHECKPOINT_SCHEMA_VERSION = "exp3-checkpoint/v1"


class Experiment3CheckpointError(ValueError):
    """Raised when checkpoint data are missing, malformed, or incompatible."""


@dataclass(frozen=True)
class RunIdentity:
    """Identity fields that must match for checkpoint resume."""

    config_hash: str
    calibration_bundle_hash: str
    source_commit_sha: str


def stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write JSON via temp file, fsync, and atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def checkpoint_payload(
    *,
    identity: RunIdentity,
    task: str,
    arm: str,
    seed: int,
    partition_hash: str,
    round_id: int,
    client_id: str | None,
    status: str,
    data: Mapping[str, object],
) -> dict[str, object]:
    """Create a checkpoint payload with exact execution identity."""
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "identity": {
            "config_hash": identity.config_hash,
            "calibration_bundle_hash": identity.calibration_bundle_hash,
            "source_commit_sha": identity.source_commit_sha,
        },
        "task": task,
        "arm": arm,
        "seed": int(seed),
        "partition_hash": partition_hash,
        "round": int(round_id),
        "client_id": client_id,
        "status": status,
        "data": dict(data),
    }


def write_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically persist a checkpoint after payload validation."""
    _validate_checkpoint_shape(payload)
    atomic_write_json(path, payload)


def _validate_checkpoint_shape(payload: Mapping[str, object]) -> None:
    required = [
        "schema_version",
        "identity",
        "task",
        "arm",
        "seed",
        "partition_hash",
        "round",
        "status",
        "data",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise Experiment3CheckpointError(f"Checkpoint is missing {missing}.")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise Experiment3CheckpointError("Unsupported checkpoint schema version.")
    identity = payload["identity"]
    if not isinstance(identity, Mapping):
        raise Experiment3CheckpointError("Checkpoint identity is malformed.")
    for key in ("config_hash", "calibration_bundle_hash", "source_commit_sha"):
        if not isinstance(identity.get(key), str) or not identity[key]:
            raise Experiment3CheckpointError(f"Checkpoint identity missing {key}.")


def read_checkpoint(path: Path, *, expected_identity: RunIdentity) -> dict[str, object]:
    """Read and validate a checkpoint for compatible explicit resume."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Experiment3CheckpointError("Checkpoint is missing or malformed.") from exc
    if not isinstance(payload, Mapping):
        raise Experiment3CheckpointError("Checkpoint root must be an object.")
    _validate_checkpoint_shape(payload)
    identity = payload["identity"]
    expected = {
        "config_hash": expected_identity.config_hash,
        "calibration_bundle_hash": expected_identity.calibration_bundle_hash,
        "source_commit_sha": expected_identity.source_commit_sha,
    }
    if dict(identity) != expected:
        raise Experiment3CheckpointError("Checkpoint belongs to an incompatible run.")
    return dict(payload)

