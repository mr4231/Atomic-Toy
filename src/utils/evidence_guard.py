"""Write protection for the frozen, canonical evidence tree.

The project's existing numeric evidence lives in a small set of study
directories under ``results/``.  Every one of those files is preserved
byte-for-byte in ``evidence_frozen/2026-09-22/`` and must never be regenerated in
place (see ``docs/PROVENANCE.md`` and ``DECISION_LOG.md``).

New experiments therefore write to a *new* output root.  This module turns that
rule into an enforced check: a writer raises :class:`ProtectedEvidenceError`
instead of silently overwriting a canonical result.  Nothing here changes
physics, parameters, dtypes or file formats - it only decides where bytes may
land.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_EVIDENCE_ROOT = ROOT / "results"

# Study directories that hold frozen, already-classified evidence.  These are the
# directories that existed when the snapshot was taken; see docs/PROVENANCE.md.
FROZEN_STUDY_DIRS = (
    "results/energy",
    "results/timestep",
    "results/dissipation",
    "results/precision",
    "results/memory",
    "results/performance",
)


class ProtectedEvidenceError(RuntimeError):
    """Raised when a write would touch frozen or already-written evidence."""


def _relative_to_root(path: Path) -> Path | None:
    """Return ``path`` relative to the repository root, or ``None`` if outside it."""
    try:
        return path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return None


def matched_frozen_dir(path: str | Path) -> str | None:
    """Return the frozen study directory containing ``path``, if any."""
    relative = _relative_to_root(Path(path))
    if relative is None:
        return None
    as_posix = relative.as_posix()
    for directory in FROZEN_STUDY_DIRS:
        if as_posix == directory or as_posix.startswith(directory + "/"):
            return directory
    return None


def assert_safe_output(path: str | Path, *, allow_protected_overwrite: bool = False) -> Path:
    """Refuse a write that would touch frozen or already-written evidence.

    The rule is deliberately narrow, so scratch work and new output roots are
    unaffected:

    1. a path inside a frozen study directory is always refused, even if the file
       does not exist yet (new files must go to a new root);
    2. a path inside ``results/`` is refused when the file already exists, so a
       previous new-root run cannot be overwritten either;
    3. anything outside ``results/`` (temporary directories, ``runs/``) is allowed.

    ``allow_protected_overwrite=True`` bypasses both refusals and exists only so a
    deliberate replacement can be expressed in code; it is not exposed on any
    command line.  The resolved path is returned for convenience.
    """
    target = Path(path)
    if allow_protected_overwrite:
        return target.resolve()
    frozen = matched_frozen_dir(target)
    if frozen is not None:
        raise ProtectedEvidenceError(
            f"refusing to write {target}: {frozen}/ holds frozen, already-classified evidence. "
            "Write to a new output root (for example --output-root results/<study>_<date>) or pass "
            "allow_protected_overwrite=True if replacing it is deliberate."
        )
    relative = _relative_to_root(target)
    if relative is not None and relative.parts and relative.parts[0] == "results" and target.exists():
        raise ProtectedEvidenceError(
            f"refusing to overwrite the existing evidence file {target}. Write to a new output root "
            "or pass allow_protected_overwrite=True if replacing it is deliberate."
        )
    return target.resolve()
