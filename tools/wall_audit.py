"""Gate G2.1 (D-06): audit the clamp+flip wall rule against an exact mirror rule.

Measurement only.  ``src/physics/boundaries.py`` is **not** modified: the exact
specular rule is defined below and installed in memory for the duration of one
run, then removed.  The production steppers are used unchanged, so the only
difference between the two runs is the wall rule.

Outputs: ``results/audit_<date>/wall/``

Run::

    python -B tools/wall_audit.py --date 2026-09-22
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
    audit_output_dir, frozen_baseline_ic, load_shipping_config, provenance, write_csv,
    write_json,
)

import src.integration.methods as methods  # noqa: E402
from src.model import SimulationConfig, SimulationState  # noqa: E402
from src.physics.boundaries import reflect_walls  # noqa: E402
from src.simulation.engine import create_random_state, run_simulation  # noqa: E402
from src.utils.io import save_result  # noqa: E402


def fold_walls(positions: np.ndarray, velocities: np.ndarray, box_length: float):
    """Exact specular reflection, folded for a step that crosses several walls.

    ``floor(x / L)`` counts the crossings of that component; an odd count means the
    normal velocity is reversed and the position mirrors back through the wall
    (``x -> 2L - x``).  Mutates in place and returns, like the production rule.
    """
    for axis in range(3):
        component = positions[:, axis]
        crossings = np.floor(component / box_length).astype(np.int64)
        folded = component - crossings * box_length
        odd = (crossings % 2) != 0
        positions[:, axis] = np.where(odd, box_length - folded, folded)
        velocities[odd, axis] *= -1.0
    return positions, velocities


class WallCounter:
    """Wrap a wall rule and count out-of-range coordinates at each step end."""

    def __init__(self, rule):
        self.rule = rule
        self.events = 0
        self.overshoots: list[float] = []

    def __call__(self, positions, velocities, box_length):
        lower = positions < 0.0
        upper = positions > box_length
        if np.any(lower | upper):
            self.events += int(np.count_nonzero(lower | upper))
            self.overshoots.extend(
                np.concatenate([-positions[lower], positions[upper] - box_length]).tolist()
            )
        return self.rule(positions, velocities, box_length)

    def summary(self) -> dict:
        overshoots = np.asarray(self.overshoots) if self.overshoots else np.zeros(1)
        return {
            "wall_events": self.events,
            "mean_overshoot": float(np.mean(overshoots)),
            "max_overshoot": float(np.max(overshoots)),
        }


def one_step_cases() -> list[dict]:
    """Deterministic single-step cases that separate the clamp from exact specular."""
    return [
        {"case": "upper wall, one crossing", "x0": 9.9, "v0": 5.0, "dt": 0.1},
        {"case": "lower wall, one crossing", "x0": 0.2, "v0": -5.0, "dt": 0.1},
        {"case": "exactly two crossings", "x0": 5.0, "v0": 200.0, "dt": 0.1},
        {"case": "three crossings", "x0": 3.0, "v0": 300.0, "dt": 0.1},
    ]


def run_one_step_case(case: dict, box_length: float) -> dict:
    """Apply both rules to one analytic single-step case."""
    positions = np.array([[case["x0"], 5.0, 5.0]])
    velocities = np.array([[case["v0"], 0.0, 0.0]])
    displaced = positions + velocities * case["dt"]

    clamp_positions, clamp_velocities = reflect_walls(displaced.copy(), velocities.copy(), box_length)
    fold_positions, fold_velocities = fold_walls(displaced.copy(), velocities.copy(), box_length)
    return {
        "case": case["case"],
        "x0": case["x0"],
        "v0": case["v0"],
        "dt": case["dt"],
        "free_flight_x": float(displaced[0, 0]),
        "clamp_x": float(clamp_positions[0, 0]),
        "fold_x": float(fold_positions[0, 0]),
        "clamp_vx": float(clamp_velocities[0, 0]),
        "fold_vx": float(fold_velocities[0, 0]),
        "position_difference": float(abs(clamp_positions[0, 0] - fold_positions[0, 0])),
        "velocity_sign_agrees": bool(
            np.signbit(clamp_velocities[0, 0]) == np.signbit(fold_velocities[0, 0])
        ),
    }


def reversibility(rule, box_length: float, dt: float, steps: int) -> dict:
    """Forward ``steps`` then backward ``steps`` with pair forces switched off.

    With ``spring_constant = 0`` and ``gamma = 0`` the velocity-Verlet step is exact
    free flight, so any irreversibility comes from the wall rule alone.
    """
    config = SimulationConfig(
        number_of_particles=3, box_size=box_length, cutoff=0.8, spring_constant=0.0,
        equilibrium_distance=0.8, initial_velocity_scale=0.0, minimum_initial_separation=0.0,
        dt=dt, steps=steps, gamma=0.0, seed=0, frame_stride=1,
    )
    start = SimulationState(
        np.array([[1.0, 9.5, 0.5], [9.0, 0.5, 9.5], [5.0, 5.0, 5.0]]),
        np.array([[3.0, -4.0, 2.0], [-2.5, 3.5, -1.5], [0.0, -6.0, 4.0]]),
        np.ones(3),
    )
    original = methods.reflect_walls
    methods.reflect_walls = rule
    try:
        forward = run_simulation(config, start)
        reversed_state = forward.final_state.copy()
        reversed_state.velocities *= -1.0
        backward = run_simulation(config, reversed_state)
    finally:
        methods.reflect_walls = original
    position_error = float(np.max(np.abs(backward.final_state.positions - start.positions)))
    velocity_error = float(np.max(np.abs(backward.final_state.velocities + start.velocities)))
    return {
        "steps_forward": steps,
        "steps_backward": steps,
        "dt": dt,
        "max_position_error": position_error,
        "max_velocity_error": velocity_error,
        "is_reversible": bool(position_error <= 1e-9 and velocity_error <= 1e-9),
    }


def drift_under_rule(config: SimulationConfig, rule):
    """Run the shipping configuration once with ``rule`` installed in memory."""
    initial = create_random_state(config)
    counter = WallCounter(rule)
    original = methods.reflect_walls
    methods.reflect_walls = counter
    try:
        result = run_simulation(config, initial)
    finally:
        methods.reflect_walls = original
    return counter.summary(), initial, result


def main() -> None:
    """Run the wall audit and report the clamp-versus-specular difference."""
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, help="audit date folder (default: today)")
    args = parser.parse_args()
    output = audit_output_dir("wall", args.date)
    box_length = 10.0

    one_step_rows = [run_one_step_case(case, box_length) for case in one_step_cases()]
    reversibility_rows = [
        {"wall_rule": "clamp_flip_production", **reversibility(reflect_walls, box_length, 0.05, 20)},
        {"wall_rule": "specular_fold", **reversibility(fold_walls, box_length, 0.05, 20)},
    ]

    config = load_shipping_config()
    clamp_summary, clamp_initial, clamp_result = drift_under_rule(config, reflect_walls)
    fold_summary, fold_initial, fold_result = drift_under_rule(config, fold_walls)
    same_ic = bool(
        np.array_equal(clamp_initial.positions, fold_initial.positions)
        and np.array_equal(clamp_initial.velocities, fold_initial.velocities)
    )
    ic_matches_frozen, ic_difference = frozen_baseline_ic()
    frozen_json = ROOT / "evidence_frozen" / "2026-09-22" / "results" / "energy" / "baseline.json"
    frozen_drift = json.loads(frozen_json.read_text(encoding="utf-8"))["max_relative_energy_drift"]

    clamp_drift = float(clamp_result.max_relative_energy_drift)
    fold_drift = float(fold_result.max_relative_energy_drift)
    drift_rows = [
        {
            "wall_rule": "clamp_flip_production", "max_relative_energy_drift": clamp_drift,
            "E0": float(clamp_result.total_energy[0]), "E_final": float(clamp_result.total_energy[-1]),
            "relative_change_vs_production": 0.0, **clamp_summary,
        },
        {
            "wall_rule": "specular_fold", "max_relative_energy_drift": fold_drift,
            "E0": float(fold_result.total_energy[0]), "E_final": float(fold_result.total_energy[-1]),
            "relative_change_vs_production": (fold_drift - clamp_drift) / clamp_drift, **fold_summary,
        },
    ]

    write_csv(output / "wall_rule_one_step.csv", one_step_rows)
    write_csv(output / "wall_reversibility.csv", reversibility_rows)
    write_csv(output / "wall_drift_comparison.csv", drift_rows)
    save_result(clamp_result, output / "clamp", initial_state=clamp_initial)
    save_result(fold_result, output / "mirror", initial_state=fold_initial)
    summary = {
        "audit": "G2.1 wall rule (D-06)",
        "shipping_config": config.as_dict(),
        "parameters": {
            "box_length": config.box_size, "dt": config.dt, "steps": config.steps,
            "n_particles": config.number_of_particles, "dtype": config.dtype, "seed": config.seed,
        },
        "initial_condition_shared_between_rules": same_ic,
        "shipping_ic_matches_frozen_baseline": ic_matches_frozen,
        "shipping_ic_max_abs_difference": ic_difference,
        "frozen_baseline_max_relative_energy_drift": float(frozen_drift),
        "clamp_run_reproduces_frozen_baseline_drift": bool(clamp_drift == float(frozen_drift)),
        "one_step_cases": one_step_rows,
        "reversibility": reversibility_rows,
        "drift_comparison": drift_rows,
        "provenance": provenance(config=config, initial_state=clamp_initial, extra={"audit": "wall"}),
        "note": "measurement only; boundaries.py is unmodified and no wall rule is chosen here",
    }
    write_json(output / "wall_audit.json", summary)

    print(f"audit root: {output}")
    print(f"shipping IC == frozen baseline IC: {ic_matches_frozen} (max abs diff {ic_difference})")
    print(f"clamp drift {clamp_drift:.12e} (frozen baseline {float(frozen_drift):.12e})")
    print(f"mirror drift {fold_drift:.12e}  relative change {(fold_drift - clamp_drift) / clamp_drift:+.2%}")
    for row in one_step_rows:
        print(
            f"  {row['case']:<28} clamp=({row['clamp_x']:.3f},{row['clamp_vx']:+.1f}) "
            f"fold=({row['fold_x']:.3f},{row['fold_vx']:+.1f})"
        )
    for row in reversibility_rows:
        print(f"  reversibility {row['wall_rule']:<24} reversible={row['is_reversible']} "
              f"max_position_error={row['max_position_error']:.6e}")


if __name__ == "__main__":
    main()
