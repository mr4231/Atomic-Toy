"""Shared helpers for the Gate G2 measurement-only audit tools.

Every audit writes into a *new* dated root (``results/audit_<date>/<name>/``) and
refuses to overwrite an existing summary, so a re-run needs a new date.  Nothing
here changes physics: the helpers only load the shipping configuration, name the
output root and attach the provenance block.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import SimulationConfig  # noqa: E402
from src.utils.provenance import provenance_record  # noqa: E402

SHIPPING_CONFIG_PATH = ROOT / "data" / "default_config.json"
FROZEN_BASELINE = ROOT / "evidence_frozen" / "2026-09-22" / "results" / "energy" / "baseline.npz"


def load_shipping_config() -> SimulationConfig:
    """Load ``data/default_config.json`` (read-only; the file is never edited)."""
    return SimulationConfig(**json.loads(SHIPPING_CONFIG_PATH.read_text(encoding="utf-8")))


def audit_output_dir(name: str, date: str | None = None) -> Path:
    """Return ``results/audit_<date>/<name>``, creating it if necessary."""
    day = date or dt.date.today().isoformat()
    target = ROOT / "results" / f"audit_{day}" / name
    target.mkdir(parents=True, exist_ok=True)
    return target


def _assert_new(path: Path) -> None:
    if path.exists():
        raise SystemExit(
            f"refusing to overwrite {path}: audits are immutable - rerun with a new --date"
        )


def write_json(path: Path, payload: dict) -> Path:
    """Write one audit summary, refusing to replace an existing audit file."""
    _assert_new(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> Path:
    """Write one audit table with the same no-overwrite rule."""
    _assert_new(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def provenance(config=None, initial_state=None, extra: dict | None = None) -> dict:
    """Provenance block (git state, versions, config and initial-condition hashes)."""
    return provenance_record(config=config, initial_state=initial_state, extra=extra)


def frozen_baseline_ic() -> tuple[bool, float]:
    """Compare the shipping initial condition with the frozen baseline (read-only)."""
    import numpy as np

    from src.simulation.engine import create_random_state

    stored = np.load(FROZEN_BASELINE)["positions"][0]
    current = create_random_state(load_shipping_config()).positions
    difference = float(np.max(np.abs(stored - current)))
    return difference == 0.0, difference
