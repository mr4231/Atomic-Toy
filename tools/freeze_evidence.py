"""Phase 0 evidence freeze: snapshot the current results into a read-only copy.

Creates ``evidence_frozen/<date>/`` containing a verbatim copy of ``results/``,
``report/`` and the animation artifact, plus a SHA-256 manifest, an environment
record, the git state and the working-tree patch.

Safety rules enforced here:
* never writes inside ``results/`` (the live evidence tree is only ever read);
* refuses to run when the destination already exists, so a snapshot is immutable.

Run from the repository root, for example::

    python -B tools/freeze_evidence.py --date 2026-09-22
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COPY_DIRS = ("results", "report")
COPY_FILES = (Path("visualisation") / "baseline.gif",)


def sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of one file without loading it at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    """Run a read-only git command and return its combined output."""
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return (result.stdout + result.stderr).strip()


def tracked_files() -> set[str]:
    """Return the repository-relative paths that git currently tracks."""
    return {line.strip() for line in git("ls-files").splitlines() if line.strip()}


def environment_report() -> str:
    """Collect the environment record that every past result file was missing."""
    lines = [
        "Evidence freeze environment record",
        f"frozen_utc: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        f"working_directory: {ROOT}",
        f"platform: {platform.platform()}",
        f"processor: {platform.processor()}",
        f"cpu_count: {os.cpu_count()}",
        f"python: {sys.version.replace(chr(10), ' ')}",
        f"python_executable: {sys.executable}",
    ]
    for distribution in ("numpy", "matplotlib", "pillow", "flask"):
        try:
            lines.append(f"{distribution}: {metadata.version(distribution)}")
        except metadata.PackageNotFoundError:
            lines.append(f"{distribution}: not installed")
    compiler = shutil.which("gcc") or shutil.which("clang") or "none"
    lines.append(f"c_compiler: {compiler}")
    if compiler != "none":
        version = subprocess.run(
            [compiler, "--version"], capture_output=True, text=True, check=False
        ).stdout.splitlines()
        lines.append(f"c_compiler_version: {version[0] if version else 'unknown'}")
    return "\n".join(lines) + "\n"


def copy_evidence(destination: Path) -> list[Path]:
    """Copy the evidence trees verbatim; raise if anything already exists."""
    copied: list[Path] = []
    for name in COPY_DIRS:
        source = ROOT / name
        if not source.is_dir():
            continue
        target = destination / name
        shutil.copytree(source, target)
        copied.extend(sorted(path for path in target.rglob("*") if path.is_file()))
    for relative in COPY_FILES:
        source = ROOT / relative
        if not source.is_file():
            continue
        target = destination / relative.name
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def original_relative(destination: Path, copied: Path) -> str:
    """Map a frozen copy back to the repository-relative source it came from."""
    relative = copied.relative_to(destination)
    if relative.parts[0] in COPY_DIRS:
        return relative.as_posix()
    return f"visualisation/{relative.name}"


def write_manifest(destination: Path, copied: list[Path]) -> Path:
    """Write the SHA-256 manifest, leaving the verdict column for Phase 1."""
    tracked = tracked_files()
    manifest = destination / "MANIFEST.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "original_path", "size_bytes", "mtime_utc", "sha256",
                "git_tracked", "provenance_verdict",
            ],
        )
        writer.writeheader()
        for path in copied:
            original = original_relative(destination, path)
            writer.writerow({
                "original_path": original,
                "size_bytes": path.stat().st_size,
                "mtime_utc": dt.datetime.fromtimestamp(
                    path.stat().st_mtime, tz=dt.timezone.utc
                ).isoformat(),
                "sha256": sha256(path),
                "git_tracked": "yes" if original in tracked else "no",
                "provenance_verdict": "",
            })
    return manifest


def write_git_state(destination: Path) -> Path:
    """Record the exact revision, dirtiness and patch that produced the freeze."""
    untracked = [line[3:].strip() for line in git("status", "--porcelain").splitlines()
                 if line[:2] == "??"]
    lines = [
        "Git state at freeze time (before the freeze commit)",
        f"HEAD: {git('rev-parse', 'HEAD')}",
        f"origin/main: {git('rev-parse', 'origin/main')}",
        f"branch: {git('rev-parse', '--abbrev-ref', 'HEAD')}",
        f"tags: {git('tag') or '(none)'}",
        "",
        "git status --porcelain:",
        git("status", "--porcelain") or "(clean)",
        "",
        "untracked file hashes:",
    ]
    for relative in sorted(untracked):
        path = ROOT / relative
        if path.is_dir():
            lines.append(f"{sha256_of_tree(path)}  {relative}/ (directory)")
        elif path.is_file():
            lines.append(f"{sha256(path)}  {relative}")
    output = destination / "GIT_STATE.txt"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (destination / "working_tree.patch").write_text(
        git("diff") + "\n", encoding="utf-8"
    )
    return output


def sha256_of_tree(directory: Path) -> str:
    """Hash a directory by hashing its sorted relative paths and contents."""
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


README_TEXT = """# Frozen evidence snapshot

This directory is an immutable copy of the project's experiment evidence, taken
before any repair work started.

* `results/`      - verbatim copy of the live `results/` tree
* `report/`       - verbatim copy of the report tree and figures
* `baseline.gif`  - the committed animation artifact
* `MANIFEST.csv`  - size, modification time and SHA-256 of every frozen file
* `GIT_STATE.txt` - revision, dirtiness, untracked hashes, and the pre-commit patch
* `working_tree.patch` - the uncommitted diff at freeze time

Rules for everything that follows:

1. Nothing in this directory may be modified, regenerated or deleted.
2. New runs write to a **new output root** (for example
   `python -m src.simulation.experiments timestep --output-root results/timestep_s2`)
   and never into `results/`.
3. `data/default_config.json` is frozen as-is; new studies get their own
   versioned copy under `data/configs/`.
4. Any file listed in `MANIFEST.csv` with `provenance_verdict` =
   `LEGACY_IC_MISMATCH` is not reproducible with the current code and must not be
   quoted elsewhere as current evidence. See `docs/PROVENANCE.md`.
5. Do not check out another branch while `main` holds the only copy of this
   work: `git checkout main` from a freeze branch would remove the files that
   exist only on that branch.
"""


def main() -> Path:
    """Create the immutable snapshot and print a short summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=dt.date.today().isoformat())
    parser.add_argument("--destination", default=None)
    args = parser.parse_args()

    destination = Path(args.destination) if args.destination else ROOT / "evidence_frozen" / args.date
    destination = destination.resolve()
    if destination.exists():
        raise SystemExit(
            f"refusing to overwrite the existing freeze at {destination}; "
            "a frozen snapshot is immutable"
        )
    destination.mkdir(parents=True)
    copied = copy_evidence(destination)
    manifest = write_manifest(destination, copied)
    git_state = write_git_state(destination)
    (destination / "ENVIRONMENT.txt").write_text(environment_report(), encoding="utf-8")
    (destination / "README.md").write_text(README_TEXT, encoding="utf-8")
    total = sum(path.stat().st_size for path in copied)
    print(f"frozen {len(copied)} files ({total} bytes) into {destination}")
    print(f"manifest: {manifest}")
    print(f"git state: {git_state}")
    return destination


if __name__ == "__main__":
    main()
