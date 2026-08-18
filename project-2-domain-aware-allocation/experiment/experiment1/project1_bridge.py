"""Utilities for reusing Project 1 without copying its runtime modules."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT2_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT2_ROOT.parent
PROJECT1_ROOT = REPO_ROOT / "project-1-adaptive-rank"

_PROJECT1_TOP_LEVEL_MODULES = {
    "config",
    "Federated",
    "Source",
    "rank_allocation",
}


def activate_project1_imports() -> Path:
    """Put Project 1 first on ``sys.path`` so its absolute imports resolve.

    Project 1 modules use imports such as ``from config import ...``. Project 2
    has some similarly named placeholders, so this function must run before any
    import of ``config``, ``Federated``, ``Source``, or ``rank_allocation``.
    """
    if not PROJECT1_ROOT.exists():
        raise FileNotFoundError(f"Project 1 root not found: {PROJECT1_ROOT}")

    for name in _PROJECT1_TOP_LEVEL_MODULES:
        module = sys.modules.get(name)
        if module is None:
            continue
        module_file = getattr(module, "__file__", None)
        if module_file and PROJECT1_ROOT in Path(module_file).resolve().parents:
            continue
        raise RuntimeError(
            "Project 1 import bridge must be activated before loading "
            f"conflicting module '{name}'. Restart the process and call "
            "activate_project1_imports() first."
        )

    p1 = str(PROJECT1_ROOT)
    if p1 in sys.path:
        sys.path.remove(p1)
    sys.path.insert(0, p1)
    return PROJECT1_ROOT

