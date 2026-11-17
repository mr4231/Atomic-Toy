"""Phase 1 provenance checker: which stored results are reproducible today?

For every stored experiment this script decides, without writing anything into
``results/``:

* ``IC_MATCH``    - the recorded configuration regenerates the stored initial
                    positions bit-for-bit, so the initial-condition generator is
                    still the one in the repository;
* ``RERUN_MATCH`` - re-integrating that configuration in memory reproduces the
                    stored trajectory arrays within ``--tolerance``;
* ``PARITY_ONLY`` - timings are machine dependent, but the recorded force-parity
                    norms and the benchmark inputs are recomputed here;
* ``FORMULA_MATCH`` - deterministic arithmetic artifacts are recomputed exactly.

Results are printed and written to ``evidence_frozen/<date>/PROVENANCE.csv``.
Usage from the repository root::

    python -B -m tools.check_provenance --date 2026-09-22
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.model import SimulationConfig  # noqa: E402
from src.physics.cell_list import pair_forces_cell_list  # noqa: E402
from src.physics.forces import pair_forces_loop, pair_forces_vectorized  # noqa: E402
from src.simulation.engine import create_random_state, run_simulation  # noqa: E402
from src.utils.metrics import pairwise_storage_bytes  # noqa: E402

FIELDS = [
    "artifact", "study", "n_particles", "cutoff", "dt", "steps", "seed", "dtype",
    "ic_max_abs_diff", "expected_frames", "rerun_frames", "positions_max_abs_diff",
    "energy_max_abs_diff", "stored_drift", "rerun_drift", "stored_e0", "rerun_e0",
    "unreproducible_metadata", "verdict", "notes",
]


def read_config(stem: Path) -> dict:
    return json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))["config"]


def missing_metadata(recorded: dict) -> str:
    expected = set(SimulationConfig(number_of_particles=1).as_dict())
    return ",".join(sorted(expected - set(recorded)))


def evaluate_run(stem: Path, tolerance: float) -> dict:
    recorded = read_config(stem)
    arrays = np.load(stem.with_suffix(".npz"))
    stored_positions = arrays["positions"]
    row = {field: "" for field in FIELDS}
    row["artifact"] = stem.name
    row["study"] = stem.parent.name
    row["n_particles"] = int(stored_positions.shape[1])
    row["expected_frames"] = int(stored_positions.shape[0])
    row["unreproducible_metadata"] = missing_metadata(recorded) or "none"

    try:
        config = SimulationConfig(**recorded)
    except (TypeError, ValueError) as error:
        row["verdict"] = "UNREADABLE_CONFIG"
        row["notes"] = str(error)
        return row

    row["cutoff"] = config.cutoff
    row["dt"] = config.dt
    row["steps"] = config.steps
    row["seed"] = config.seed
    row["dtype"] = config.dtype

    state = create_random_state(config)
    row["ic_max_abs_diff"] = float(
        np.max(np.abs(state.positions.astype("float64") - stored_positions[0].astype("float64")))
    )

    result = run_simulation(config, state)
    row["rerun_frames"] = int(result.positions.shape[0])
    shared = min(result.positions.shape[0], stored_positions.shape[0])
    row["positions_max_abs_diff"] = float(
        np.max(np.abs(result.positions[:shared].astype("float64") - stored_positions[:shared].astype("float64")))
    )
    row["energy_max_abs_diff"] = float(
        np.max(np.abs(
            result.total_energy[:shared].astype("float64") - arrays["total_energy"][:shared].astype("float64")
        ))
    )
    row["stored_drift"] = float(
        json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))["max_relative_energy_drift"]
    )
    row["rerun_drift"] = result.max_relative_energy_drift
    row["stored_e0"] = float(arrays["total_energy"][0])
    row["rerun_e0"] = float(result.total_energy[0])

    ic_ok = row["ic_max_abs_diff"] <= tolerance
    if not ic_ok:
        row["verdict"] = "LEGACY_IC_MISMATCH"
        row["notes"] = (
            "recorded configuration does not regenerate the stored initial positions; "
            "no stored initial-condition state exists, so the run cannot be replayed"
        )
    elif row["positions_max_abs_diff"] <= tolerance:
        row["verdict"] = "REPRODUCIBLE"
        row["notes"] = "initial conditions and trajectory reproduce"
    else:
        row["verdict"] = "RERUN_MISMATCH"
        row["notes"] = "initial conditions match but the trajectory does not"
    return row


def evaluate_benchmark(path: Path, tolerance: float) -> list[dict]:
    """Recompute the recorded force-parity norms; timings stay machine dependent."""
    with path.open(encoding="utf-8", newline="") as stream:
        recorded_rows = list(csv.DictReader(stream))
    cache: dict[int, dict[str, np.ndarray]] = {}
    rows: list[dict] = []
    for recorded in recorded_rows:
        n_particles = int(recorded["n_particles"])
        if n_particles not in cache:
            positions = np.random.default_rng(123 + n_particles).uniform(0.0, 10.0, (n_particles, 3))
            reference = pair_forces_loop(positions, 2.5, 15.0, 0.8)
            cache[n_particles] = {
                "python_loop": reference,
                "numpy_vectorized": pair_forces_vectorized(positions, 2.5, 15.0, 0.8),
                "cell_list_spatial": pair_forces_cell_list(positions, 10.0, 2.5, 15.0, 0.8),
            }
        row = {field: "" for field in FIELDS}
        row["artifact"] = path.name
        row["study"] = "performance"
        row["n_particles"] = n_particles
        row["cutoff"] = 2.5
        implementation = recorded["implementation"]
        expected = float(recorded["force_parity_norm"])
        if implementation == "c_optimized":
            row["verdict"] = "NOT_CHECKED"
            row["notes"] = "C library deliberately not rebuilt by this read-only check"
        else:
            parity = float(np.linalg.norm(cache[n_particles]["python_loop"] - cache[n_particles][implementation]))
            row["positions_max_abs_diff"] = parity
            row["verdict"] = "PARITY_ONLY" if abs(parity - expected) <= tolerance else "PARITY_MISMATCH"
            row["notes"] = (
                f"{implementation}: recorded parity {expected:.3e}, recomputed {parity:.3e}; "
                "wall-clock column is machine dependent and was not re-timed"
            )
        rows.append(row)
    return rows


def evaluate_memory(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        recorded_rows = list(csv.DictReader(stream))
    rows: list[dict] = []
    for recorded in recorded_rows:
        n_particles = int(recorded["n_particles"])
        dtype = recorded["dtype"]
        recomputed = pairwise_storage_bytes(n_particles, dtype)
        expected = int(recorded["pair_displacement_bytes"])
        row = {field: "" for field in FIELDS}
        row["artifact"] = path.name
        row["study"] = "memory"
        row["n_particles"] = n_particles
        row["dtype"] = dtype
        row["positions_max_abs_diff"] = abs(recomputed - expected)
        row["verdict"] = "FORMULA_MATCH" if recomputed == expected else "FORMULA_MISMATCH"
        row["notes"] = f"recomputed {recomputed} bytes vs recorded {expected} bytes"
        rows.append(row)
    return rows


def discover_runs(root: Path) -> list[Path]:
    return sorted(path.with_suffix("") for path in root.glob("**/*.npz"))


def check(root: Path, tolerance: float, skip_benchmark: bool) -> list[dict]:
    rows = [evaluate_run(stem, tolerance) for stem in discover_runs(root)]
    for candidate in sorted(root.glob("**/pairwise_memory.csv")):
        rows.extend(evaluate_memory(candidate))
    if not skip_benchmark:
        for candidate in sorted(root.glob("**/benchmark.csv")):
            rows.extend(evaluate_benchmark(candidate, tolerance))
    return rows


def write_report(rows: list[dict], destination: Path) -> Path:
    if "results" in destination.parts:
        raise SystemExit("refusing to write a provenance report inside results/")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def main() -> None:
    import datetime as dt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="evidence root to inspect (default: results/)")
    parser.add_argument("--date", default=dt.date.today().isoformat())
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve() if args.root else ROOT / "results"
    rows = check(root, args.tolerance, args.skip_benchmark)
    destination = ROOT / "evidence_frozen" / args.date / "PROVENANCE.csv"
    if not destination.parent.exists():
        destination = ROOT / "provenance" / f"PROVENANCE_{args.date}.csv"
    print(f"inspected: {root}")
    print(f"{'artifact':<22}{'study':<13}{'N':>5}{'cutoff':>8}{'dt':>8}{'frames':>8}  {'verdict':<22}notes")
    for row in rows:
        print(
            f"{row['artifact']:<22}{row['study']:<13}{str(row['n_particles']):>5}"
            f"{str(row['cutoff']):>8}{str(row['dt']):>8}{str(row['expected_frames']):>8}  "
            f"{row['verdict']:<22}{row['notes'][:80]}"
        )
    print(f"report: {write_report(rows, destination)}")


if __name__ == "__main__":
    main()
