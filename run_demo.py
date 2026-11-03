"""Run the baseline, write its data, and optionally render or view its animation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.physics.geometry import pairwise_geometry
from src.simulation.experiments import load_config
from src.simulation.engine import create_random_state, initial_state_diagnostics, run_simulation
from src.utils.evidence_guard import ProtectedEvidenceError
from src.utils.io import save_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="data/default_config.json")
    parser.add_argument("--output", default="results/energy/baseline")
    parser.add_argument("--gif", action="store_true", help="also write visualisation/baseline.gif")
    parser.add_argument("--view", action="store_true", help="open an interactive 3D animation window")
    parser.add_argument("--fps", type=int, default=25, help="animation frames per second (default: 25)")
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="interactive playback multiplier; does not change simulation data (default: 1)",
    )
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be a positive integer")
    if args.speed <= 0:
        parser.error("--speed must be positive")
    config = load_config(args.config)
    initial_state = create_random_state(config)
    initial_diagnostics = initial_state_diagnostics(initial_state, config.cutoff)
    result = run_simulation(config, initial_state)
    output = Path(args.output)
    try:
        save_result(result, output, initial_state=initial_state)
    except ProtectedEvidenceError as error:
        # results/ holds frozen evidence; an explicit message is preferable to a traceback.
        raise SystemExit(f"refused: {error}") from None
    print(f"Saved {output}.npz and {output}.json")
    print(f"Maximum relative energy drift: {result.max_relative_energy_drift:.3e}")
    print(
        "Initial diagnostics: "
        f"minimum separation={initial_diagnostics['minimum_initial_particle_separation']:.3f}, "
        f"average speed={initial_diagnostics['average_initial_speed']:.3f}, "
        f"maximum speed={initial_diagnostics['maximum_initial_speed']:.3f}, "
        f"interacting pairs={initial_diagnostics['initial_interacting_pair_count']}"
    )
    later_frame = len(result.positions) // 2
    later_geometry = pairwise_geometry(result.positions[later_frame], config.cutoff)
    later_pairs = int(np.count_nonzero(np.triu(later_geometry.neighbors, k=1)))
    print(
        f"Later diagnostics at t={result.frame_times[later_frame]:.3f}: "
        f"interacting pairs={later_pairs}"
    )
    if args.gif:
        from visualisation.animate import animate_trajectory

        print(f"Saved {animate_trajectory(output, 'visualisation/baseline.gif', args.fps)}")
    if args.view:
        from visualisation.animate import view_trajectory

        view_trajectory(output, args.fps, args.speed)


if __name__ == "__main__":
    main()
