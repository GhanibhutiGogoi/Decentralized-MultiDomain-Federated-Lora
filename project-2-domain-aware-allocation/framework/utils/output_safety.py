"""Output directory preflight checks for Project 2 experiment runners."""

from __future__ import annotations

import shutil
from pathlib import Path


class OutputDirectorySafetyError(RuntimeError):
    """Raised when an experiment output directory is unsafe to reuse."""


def _is_directory_empty(path: Path) -> bool:
    return not any(path.iterdir())


def _assert_cleanup_target(path: Path) -> None:
    resolved = path.resolve()
    if not path.exists() or not path.is_dir():
        raise OutputDirectorySafetyError(
            f"Refusing to clean non-directory output path: {path}"
        )
    if resolved.parent == resolved or resolved == Path(resolved.anchor):
        raise OutputDirectorySafetyError(
            f"Refusing to clean filesystem root as an output directory: {path}"
        )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolved_path(path: Path, label: str) -> Path:
    try:
        return Path(path).resolve()
    except (OSError, RuntimeError) as exc:
        raise OutputDirectorySafetyError(
            f"Refusing unsafe {label} path that cannot be resolved: {path}"
        ) from exc


def _is_filesystem_root(path: Path) -> bool:
    return path.parent == path or path == Path(path.anchor)


def ensure_cleanup_within_allowed_root(
    output_dir: Path,
    allowed_root: Path,
    *,
    repository_root: Path | None = None,
    project_root: Path | None = None,
    shared_outputs_root: Path | None = None,
    output_label: str = "output directory",
    allowed_label: str = "allowed output root",
) -> None:
    """Reject overwrite cleanup targets outside the experiment-owned root."""
    output_path = Path(output_dir)
    output_resolved = _resolved_path(output_path, output_label)
    allowed_resolved = _resolved_path(Path(allowed_root), allowed_label)

    if output_path.exists() and not output_path.is_dir():
        raise OutputDirectorySafetyError(
            f"{output_label} exists but is not a directory: {output_dir}"
        )
    if output_path.is_symlink() and not output_path.exists():
        raise OutputDirectorySafetyError(
            f"{output_label} is an unresolved symlink: {output_dir}"
        )
    if _is_filesystem_root(output_resolved):
        raise OutputDirectorySafetyError(
            f"{output_label} cannot be a filesystem root: {output_dir}"
        )

    forbidden_roots = [
        (repository_root, "repository root"),
        (project_root, "Project 2 root"),
        (shared_outputs_root, "shared Project 2 outputs directory"),
    ]
    for forbidden, label in forbidden_roots:
        if forbidden is not None and output_resolved == _resolved_path(
            Path(forbidden),
            label,
        ):
            raise OutputDirectorySafetyError(
                f"{output_label} cannot be the {label}: {output_dir}"
            )

    if output_resolved == allowed_resolved:
        return
    if _is_relative_to(output_resolved, allowed_resolved):
        return

    raise OutputDirectorySafetyError(
        f"{output_label} must be {allowed_label} or one of its descendants: "
        f"{output_dir} vs {allowed_root}"
    )


def ensure_disjoint_directory(
    output_dir: Path,
    protected_dir: Path,
    *,
    output_label: str = "output directory",
    protected_label: str = "protected directory",
) -> None:
    """Reject equal, nested, or parent/child-overlapping directories."""
    output_resolved = Path(output_dir).resolve()
    protected_resolved = Path(protected_dir).resolve()
    if (
        output_resolved == protected_resolved
        or _is_relative_to(output_resolved, protected_resolved)
        or _is_relative_to(protected_resolved, output_resolved)
    ):
        raise OutputDirectorySafetyError(
            f"{output_label} cannot overlap {protected_label}: "
            f"{output_dir} vs {protected_dir}"
        )


def ensure_not_cleanup_parent(
    output_dir: Path,
    protected_dir: Path,
    *,
    output_label: str = "output directory",
    protected_label: str = "protected directory",
) -> None:
    """Reject cleanup targets that are a protected directory or its parent."""
    output_resolved = Path(output_dir).resolve()
    protected_resolved = Path(protected_dir).resolve()
    if output_resolved == protected_resolved or _is_relative_to(
        protected_resolved,
        output_resolved,
    ):
        raise OutputDirectorySafetyError(
            f"{output_label} cannot be {protected_label} or one of its "
            f"parents: {output_dir} vs {protected_dir}"
        )


def _remove_child(path: Path) -> None:
    is_junction = getattr(path, "is_junction", lambda: False)
    if path.is_symlink() or is_junction() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _clean_directory_contents(path: Path) -> None:
    _assert_cleanup_target(path)
    for child in path.iterdir():
        _remove_child(child)


def prepare_output_directory(
    output_dir: Path,
    *,
    overwrite: bool = False,
    experiment_name: str = "Experiment",
    allowed_cleanup_root: Path | None = None,
    repository_root: Path | None = None,
    project_root: Path | None = None,
    shared_outputs_root: Path | None = None,
) -> Path:
    """Create or validate an output directory before experiment execution.

    Nonexistent and empty directories are accepted. Non-empty directories are
    rejected by default so stale artifacts cannot mix with fresh outputs. With
    explicit overwrite authorization, only the selected directory's contents are
    removed; unrelated siblings are not touched.
    """
    output_dir = Path(output_dir)

    if overwrite and allowed_cleanup_root is None:
        raise OutputDirectorySafetyError(
            f"{experiment_name} overwrite cleanup requires an allowed output root."
        )
    if overwrite:
        ensure_cleanup_within_allowed_root(
            output_dir,
            allowed_cleanup_root,
            repository_root=repository_root,
            project_root=project_root,
            shared_outputs_root=shared_outputs_root,
            output_label=f"{experiment_name} output directory",
            allowed_label=f"{experiment_name} allowed output root",
        )

    if output_dir.exists() and not output_dir.is_dir():
        raise OutputDirectorySafetyError(
            f"{experiment_name} output path exists but is not a directory: "
            f"{output_dir}"
        )

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    if _is_directory_empty(output_dir):
        return output_dir

    if not overwrite:
        raise OutputDirectorySafetyError(
            f"{experiment_name} output directory is non-empty: {output_dir}. "
            "Refusing to run because existing artifacts could mix with new "
            "outputs. Choose an empty --output-dir or rerun with --overwrite "
            "to clean only this experiment output directory first."
        )

    _clean_directory_contents(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
