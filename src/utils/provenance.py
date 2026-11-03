"""Reproducibility metadata attached to every newly written result file.

Nothing in this module changes physics, parameters or numerical results: the
helpers only describe *which* code, environment and initial state produced a
saved artifact, so that a future reader never again has to guess (see
``docs/PROVENANCE.md`` for the legacy artifacts that had to be classified as
unreproducible).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROVENANCE_KEY = "provenance"


def _git(*args: str) -> str:
    """Run a read-only git command inside the repository root."""
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        )
    except OSError:
        return ""
    return (result.stdout or "").strip()


def git_state() -> dict[str, Any]:
    """Return the revision, branch and dirtiness of the producing checkout."""
    status = _git("status", "--porcelain")
    return {
        "git_commit": _git("rev-parse", "HEAD") or "unknown",
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "git_dirty": bool(status),
        "git_dirty_entries": len(status.splitlines()) if status else 0,
    }


def tool_versions() -> dict[str, str]:
    """Return the interpreter and library versions that can change a result."""
    versions = {"python": platform.python_version()}
    for distribution in ("numpy", "matplotlib", "pillow", "flask"):
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = "not installed"
    return versions


def array_fingerprint(*arrays: np.ndarray | None) -> str | None:
    """SHA-256 over dtype, shape and raw bytes; ``None`` when any array is missing."""
    if not arrays or any(array is None for array in arrays):
        return None
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(str(contiguous.shape).encode("utf-8"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def initial_condition_fingerprint(state: Any) -> str | None:
    """Fingerprint a ``SimulationState`` (duck typed) so two runs can be compared."""
    return array_fingerprint(
        getattr(state, "positions", None),
        getattr(state, "velocities", None),
        getattr(state, "masses", None),
    )


def config_fingerprint(config: Any) -> str | None:
    """SHA-256 over the canonical JSON of a configuration (duck typed).

    The digest depends only on the recorded parameter values, so two runs can be
    compared without diffing whole JSON files.
    """
    if config is None:
        return None
    as_dict = getattr(config, "as_dict", None)
    values = as_dict() if callable(as_dict) else dict(config)
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def provenance_record(
    *,
    initial_state: Any = None,
    config: Any = None,
    command: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the provenance block stored next to every new result."""
    record: dict[str, Any] = {
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command_line": command if command is not None else " ".join(sys.argv),
        "platform": platform.platform(),
        "initial_condition_sha256": initial_condition_fingerprint(initial_state),
        "config_sha256": config_fingerprint(config),
    }
    record.update(git_state())
    record.update(tool_versions())
    if extra:
        record.update(extra)
    return record
