
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
    return SimulationConfig(**json.loads(SHIPPING_CONFIG_PATH.read_text(encoding="utf-8")))


def audit_output_dir(name: str, date: str | None = None) -> Path:
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
    _assert_new(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> Path:
    _assert_new(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def provenance(config=None, initial_state=None, extra: dict | None = None) -> dict:
    return provenance_record(config=config, initial_state=initial_state, extra=extra)


def frozen_baseline_ic() -> tuple[bool, float]:
    import numpy as np

    from src.simulation.engine import create_random_state

    stored = np.load(FROZEN_BASELINE)["positions"][0]
    current = create_random_state(load_shipping_config()).positions
    difference = float(np.max(np.abs(stored - current)))
    return difference == 0.0, difference
