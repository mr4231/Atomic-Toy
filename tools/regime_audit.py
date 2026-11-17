"""Gate G2.3 (D-09): quantify the interaction regime of the shipping configuration.

Measurement only.  ``data/default_config.json`` is **not** modified: the activity
sweep evaluates alternative (N, initial_velocity_scale) combinations in memory and
writes only tables into the new audit root.

Outputs: ``results/audit_<date>/regime/``

Run::

    python -B tools/regime_audit.py --date 2026-09-22
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

import src.integration.methods as methods  # noqa: E402
from src.simulation.engine import create_random_state, run_simulation  # noqa: E402

# Small, explicitly declared activity sweep (audit probes only - never written
# back into any configuration file).
SWEEP = (
    {"N": 25, "sigma": 8.0},   # shipping values
    {"N": 50, "sigma": 8.0},
    {"N": 25, "sigma": 2.0},
    {"N": 50, "sigma": 2.0},
)


class WallEventCounter:
    """Wrap the wall rule the steppers actually call and count events per call.

    Note: ``src/integration/methods.py`` binds ``reflect_walls`` into its own
    namespace at import time, so the patch must be applied to
    ``src.integration.methods.reflect_walls`` (as the wall audit also does),
    not to ``src.physics.boundaries``.
    """

    def __init__(self):
        self.events = 0
        self._original = methods.reflect_walls

    def __call__(self, positions, velocities, box_length):
        lower = positions < 0.0
        upper = positions > box_length
        self.events += int(np.count_nonzero(lower | upper))
        return self._original(positions, velocities, box_length)

    def install(self):
        methods.reflect_walls = self
        return self

    def uninstall(self):
        methods.reflect_walls = self._original


def regime_metrics(config, state) -> dict:
    counter = WallEventCounter().install()
    try:
        result = run_simulation(config, state)
    finally:
        counter.uninstall()

    initial = state.positions
    displacement = initial[:, None, :] - initial[None, :, :]
    distance = np.linalg.norm(displacement, axis=2)
    neighbours_initial = distance <= config.cutoff
    np.fill_diagonal(neighbours_initial, False)
    initial_pairs = int(np.count_nonzero(neighbours_initial) // 2)

    energy = result.total_energy
    e0 = float(energy[0])
    return {
        "N": config.number_of_particles,
        "box_size": config.box_size,
        "cutoff": config.cutoff,
        "equilibrium_distance": config.equilibrium_distance,
        "spring_constant": config.spring_constant,
        "initial_velocity_scale": config.initial_velocity_scale,
        "dt": config.dt,
        "steps": config.steps,
        "gamma": config.gamma,
        "dtype": config.dtype,
        "seed": config.seed,
        "E0": e0,
        "U0": float(result.potential[0]),
        "U_max": float(np.max(result.potential)),
        "U_max_over_E0": float(np.max(result.potential)) / abs(e0),
        "K0_over_E0": float(result.kinetic[0]) / abs(e0),
        "max_relative_energy_drift": float(result.max_relative_energy_drift),
        "initial_interacting_pairs": initial_pairs,
        "mean_speed_initial": float(np.mean(np.linalg.norm(state.velocities, axis=1))),
        "wall_events": counter.events,
        "wall_events_per_particle": counter.events / config.number_of_particles,
        "frames": int(result.positions.shape[0]),
    }


def pair_event_rate(config, state) -> dict:
    result = run_simulation(config, state)
    previous = None
    entries = exits = 0
    pair_counts = []
    for positions in result.positions:
        displacement = positions[:, None, :] - positions[None, :, :]
        distance = np.linalg.norm(displacement, axis=2)
        neighbours = distance <= config.cutoff
        np.fill_diagonal(neighbours, False)
        pairs = frozenset(
            tuple(sorted((int(i), int(j))))
            for i, j in zip(*np.nonzero(np.triu(neighbours)))
        )
        pair_counts.append(len(pairs))
        if previous is not None:
            entries += len(pairs - previous)
            exits += len(previous - pairs)
        previous = pairs
    return {
        "mean_interacting_pairs": float(np.mean(pair_counts)),
        "max_interacting_pairs": int(np.max(pair_counts)),
        "pair_entry_events": entries,
        "pair_exit_events": exits,
        "frames_analysed": int(result.positions.shape[0]),
        "pair_events_per_particle_per_unit_time": (entries + exits)
        / (config.number_of_particles * float(result.times[-1])),
    }


def probe_config(shipping, case):
    fields = shipping.as_dict()
    for legacy in ("n_particles", "initial_velocity"):
        fields.pop(legacy, None)
    fields["number_of_particles"] = case["N"]
    fields["initial_velocity_scale"] = case["sigma"]
    return type(shipping)(**fields)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, help="audit date folder (default: today)")
    args = parser.parse_args()
    output = audit_output_dir("regime", args.date)

    shipping = load_shipping_config()
    shipping_state = create_random_state(shipping)

    shipping_row = regime_metrics(shipping, shipping_state)
    shipping_row.update(pair_event_rate(shipping, shipping_state))

    sweep_rows = [shipping_row]
    for case in SWEEP[1:]:
        probe = probe_config(shipping, case)
        state = create_random_state(probe)
        row = regime_metrics(probe, state)
        row.update(pair_event_rate(probe, state))
        sweep_rows.append(row)

    write_csv(output / "regime_summary.csv", sweep_rows)
    write_json(output / "regime_audit.json", {
        "audit": "G2.3 interaction regime (D-09)",
        "shipping_config": shipping.as_dict(),
        "shipping_regime": shipping_row,
        "sweep_cases_declared": SWEEP,
        "sweep": sweep_rows,
        "provenance": provenance(config=shipping, initial_state=shipping_state,
                                 extra={"audit": "regime"}),
        "note": ("measurement only; data/default_config.json is unmodified and no "
                 "final physics regime is chosen here"),
    })

    print(f"audit root: {output}")
    for row in sweep_rows:
        marker = " <- shipping" if (row["N"] == 25 and row["initial_velocity_scale"] == 8.0) else ""
        print(f"  N={row['N']:<4} sigma={row['initial_velocity_scale']:<5} "
              f"U_max/E0={row['U_max_over_E0']:.3e} "
              f"pairs mean/max={row['mean_interacting_pairs']:.2f}/{row['max_interacting_pairs']} "
              f"pair_ev/part/t={row['pair_events_per_particle_per_unit_time']:.3f} "
              f"wall_ev={row['wall_events']} drift={row['max_relative_energy_drift']:.3e}"
              f"{marker}")


if __name__ == "__main__":
    main()
