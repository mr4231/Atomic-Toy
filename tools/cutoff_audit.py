"""Gate G2.2 (D-08): audit U and F continuity at the cutoff, and a pair crossing.

Measurement only.  ``forces.py``, ``energy.py`` and ``data/default_config.json``
are **not** modified: the candidate cutoffs are passed as arguments to the existing
production functions, and every candidate configuration is evaluated in memory.

Outputs: ``results/audit_<date>/cutoff/``

Run::

    python -B tools/cutoff_audit.py --date 2026-09-22
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit_common import (  # noqa: E402
    audit_output_dir, load_shipping_config, provenance, write_csv, write_json,
)

from src.model import SimulationConfig, SimulationState  # noqa: E402
from src.physics.energy import potential_energy  # noqa: E402
from src.physics.forces import pair_forces_loop, pair_forces_vectorized  # noqa: E402
from src.simulation.engine import run_simulation  # noqa: E402
from src.utils.provenance import array_fingerprint  # noqa: E402

# Candidate (cutoff, equilibrium_distance) pairs.  r_cut = r0 is the shipping
# configuration; the others are audit probes only and are never written to config.
CANDIDATES = ((0.8, 0.8), (1.2, 0.8), (2.5, 0.8))
PROBE = 1e-9          # distance offset used to approach the cutoff from each side
# Audit threshold: the value just *inside* the cutoff counts as zero (continuous)
# when it is at most this.  It is a reporting convention, not a physics decision.
CONTINUITY_EPSILON = 1e-6

CROSSING_CASES = (
    {"cutoff": 0.8, "r0": 0.8, "r_start": 0.75, "v_rel": 0.2},
    {"cutoff": 1.2, "r0": 0.8, "r_start": 1.10, "v_rel": 0.2},
    {"cutoff": 2.5, "r0": 0.8, "r_start": 2.40, "v_rel": 0.5},
)
DTS = (0.05, 0.02, 0.01, 0.005)
HORIZON = 1.0
BOX = 50.0


def pair_at(radius: float) -> np.ndarray:
    """Two particles separated by ``radius`` along x, centred away from any wall."""
    return np.array([[0.0, 0.0, 0.0], [radius, 0.0, 0.0]])


def continuity_rows(spring_constant: float) -> list[dict]:
    """Evaluate the production U and F just inside and just outside each cutoff."""
    rows: list[dict] = []
    for cutoff, equilibrium in CANDIDATES:
        for side, radius in (("below", cutoff - PROBE), ("above", cutoff + PROBE)):
            positions = pair_at(radius)
            vectorized = pair_forces_vectorized(positions, cutoff, spring_constant, equilibrium)
            loop = pair_forces_loop(positions, cutoff, spring_constant, equilibrium)
            rows.append({
                "cutoff": cutoff,
                "equilibrium_distance": equilibrium,
                "side": side,
                "r": radius,
                "potential_energy": potential_energy(positions, cutoff, spring_constant, equilibrium),
                "force_on_first_particle_x": float(vectorized[0, 0]),
                "force_magnitude": float(abs(vectorized[0, 0])),
                "loop_equals_vectorized": bool(np.array_equal(loop, vectorized)),
            })
    return rows


def continuity_summary(rows: list[dict], spring_constant: float) -> list[dict]:
    """Reduce the probe rows to per-candidate continuity deltas."""
    summary: list[dict] = []
    for cutoff, equilibrium in CANDIDATES:
        pair = [row for row in rows if row["cutoff"] == cutoff]
        below = next(row for row in pair if row["side"] == "below")
        above = next(row for row in pair if row["side"] == "above")
        potential_jump = below["potential_energy"] - above["potential_energy"]
        force_jump = below["force_magnitude"] - above["force_magnitude"]
        summary.append({
            "cutoff": cutoff,
            "equilibrium_distance": equilibrium,
            "cutoff_minus_equilibrium": cutoff - equilibrium,
            "potential_below": below["potential_energy"],
            "potential_above": above["potential_energy"],
            "potential_jump": potential_jump,
            "force_magnitude_below": below["force_magnitude"],
            "force_magnitude_above": above["force_magnitude"],
            "force_jump": force_jump,
            "analytic_force_below": spring_constant * PROBE,
            "potential_continuous_within_probe": bool(below["potential_energy"] <= CONTINUITY_EPSILON),
            "force_continuous_within_probe": bool(below["force_magnitude"] <= CONTINUITY_EPSILON),
        })
    return summary


def crossing_run(case: dict, dt: float, spring_constant: float) -> dict:
    """Integrate one isolated pair outward through its own cutoff."""
    config = SimulationConfig(
        number_of_particles=2, box_size=BOX, cutoff=case["cutoff"], spring_constant=spring_constant,
        equilibrium_distance=case["r0"], initial_velocity_scale=0.0, minimum_initial_separation=0.0,
        dt=dt, steps=round(HORIZON / dt), gamma=0.0, seed=0, frame_stride=1,
    )
    separation = case["r_start"]
    state = SimulationState(
        np.array([
            [BOX / 2 - separation / 2, BOX / 2, BOX / 2],
            [BOX / 2 + separation / 2, BOX / 2, BOX / 2],
        ]),
        np.array([[-case["v_rel"] / 2, 0.0, 0.0], [case["v_rel"] / 2, 0.0, 0.0]]),
        np.ones(2),
    )
    result = run_simulation(config, state)
    radius = np.linalg.norm(result.positions[:, 1] - result.positions[:, 0], axis=1)
    outside = radius > case["cutoff"]
    crossing = int(np.argmax(outside)) if bool(np.any(outside)) else -1
    energy = result.total_energy
    jump = float(energy[crossing] - energy[crossing - 1]) if crossing > 0 else None
    return {
        "cutoff": case["cutoff"],
        "equilibrium_distance": case["r0"],
        "cutoff_minus_equilibrium": case["cutoff"] - case["r0"],
        "dt": dt,
        "steps": config.steps,
        "r_start": case["r_start"],
        "v_rel": case["v_rel"],
        "E0": float(energy[0]),
        "max_relative_energy_drift": float(np.max(np.abs(energy - energy[0])) / abs(energy[0])),
        "crossed_within_horizon": crossing > 0,
        "t_crossing": float(result.times[crossing]) if crossing > 0 else None,
        "r_before": float(radius[crossing - 1]) if crossing > 0 else None,
        "r_after": float(radius[crossing]) if crossing > 0 else None,
        "energy_before": float(energy[crossing - 1]) if crossing > 0 else None,
        "energy_after": float(energy[crossing]) if crossing > 0 else None,
        "energy_jump_at_crossing": jump,
        "relative_energy_jump": abs(jump) / abs(float(energy[0])) if jump is not None else None,
        "wall_contacts": int(np.count_nonzero((result.positions <= 0.0) | (result.positions >= BOX))),
        "initial_condition_sha256": array_fingerprint(state.positions, state.velocities, state.masses),
    }


def main() -> None:
    """Run the cutoff continuity probe and the single-pair crossing experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, help="audit date folder (default: today)")
    args = parser.parse_args()
    output = audit_output_dir("cutoff", args.date)

    shipping = load_shipping_config()
    spring = shipping.spring_constant
    rows = continuity_rows(spring)
    summary = continuity_summary(rows, spring)
    crossing = [crossing_run(case, dt, spring) for case in CROSSING_CASES for dt in DTS]

    write_csv(output / "cutoff_continuity_probe.csv", rows)
    write_csv(output / "cutoff_continuity_summary.csv", summary)
    write_csv(output / "cutoff_crossing.csv", crossing)
    write_json(output / "cutoff_audit.json", {
        "audit": "G2.2 cutoff continuity and single-pair crossing (D-08)",
        "shipping_config": shipping.as_dict(),
        "probe_offset": PROBE,
        "continuity_epsilon_threshold": CONTINUITY_EPSILON,
        "horizon": HORIZON,
        "box_size": BOX,
        "continuity_probe": rows,
        "continuity_summary": summary,
        "crossing": crossing,
        "provenance": provenance(config=shipping, extra={"audit": "cutoff"}),
        "note": ("measurement only; forces.py, energy.py and data/default_config.json are "
                 "unmodified and no cutoff is chosen here"),
    })

    print(f"audit root: {output}")
    print(f"production spring_constant={spring}, equilibrium_distance={shipping.equilibrium_distance}")
    print("continuity just inside / outside the cutoff (probe = 1e-9):")
    for row in summary:
        print(f"  r_cut={row['cutoff']:<5} r0={row['equilibrium_distance']:<5} "
              f"potential_jump={row['potential_jump']:.6e} force_jump={row['force_jump']:.6e} "
              f"U_continuous={row['potential_continuous_within_probe']} "
              f"F_continuous={row['force_continuous_within_probe']}")
    print("single-pair crossing (outward, no wall contact expected):")
    for row in crossing:
        print(f"  r_cut={row['cutoff']:<5} dt={row['dt']:<7} t_cross={row['t_crossing']} "
              f"energy_jump={row['energy_jump_at_crossing']} "
              f"rel_jump={row['relative_energy_jump']} "
              f"max_drift={row['max_relative_energy_drift']:.3e} walls={row['wall_contacts']}")


if __name__ == "__main__":
    main()
