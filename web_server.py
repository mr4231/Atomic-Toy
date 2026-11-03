"""Flask Web Application Server for the Atomic Toy Model Dashboard.

Connects the web presentation layer directly to the existing Python simulation backend
(src.simulation, src.physics, src.integration, src.dissipation) without rewriting physics logic.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

from src.model import SimulationConfig
from src.physics.geometry import pairwise_geometry
from src.simulation.engine import create_random_state, initial_state_diagnostics, run_simulation
from src.simulation.experiments import (
    gamma_sweep, load_config, memory_table, precision_comparison, timestep_sweep,
)
from src.simulation.benchmarks import benchmark_forces, benchmark_protocol, save_benchmark_csv

app = Flask(__name__, template_folder="templates", static_folder="static")
ROOT_DIR = Path(__file__).resolve().parent

# results/ is frozen, canonical evidence and is read-only from the web layer.
CANONICAL_EVIDENCE_DIR = ROOT_DIR / "results"
# Every web-triggered experiment writes to a fresh directory here instead.
WEB_RUN_ROOT = ROOT_DIR / "runs"
VALID_EXPERIMENTS = ("timestep", "gamma", "precision", "memory", "performance")
MAX_INTERACTIVE_PARTICLES = 200
MAX_INTERACTIVE_STEPS = 5_000
MAX_INTERACTIVE_FRAMES = 1_000


def assert_outside_canonical_evidence(target: Path) -> Path:
    """Raise when a write would land inside the frozen evidence tree."""
    resolved = target.resolve()
    canonical = CANONICAL_EVIDENCE_DIR.resolve()
    if resolved == canonical or canonical in resolved.parents:
        raise ValueError(f"refusing to write inside the canonical evidence tree: {resolved}")
    return resolved


def new_run_directory(experiment: str) -> Path:
    """Return a fresh per-request output root that cannot overwrite an existing run."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    target = assert_outside_canonical_evidence(WEB_RUN_ROOT / f"{experiment}_{stamp}")
    if target.exists():
        raise FileExistsError(f"run directory already exists: {target}")
    return target


@app.route("/")
def index():
    """Render the main scientific dashboard home page."""
    return render_template("index.html")


@app.route("/api/config/default", methods=["GET"])
def get_default_config():
    """Return default configuration from data/default_config.json."""
    config_path = ROOT_DIR / "data" / "default_config.json"
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return jsonify(data)
    return jsonify(SimulationConfig().as_dict())


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """Run interactive 3D simulation with caller-specified parameters."""
    try:
        req_data = request.get_json(force=True) or {}
        # Parse parameters into SimulationConfig
        config = SimulationConfig(
            number_of_particles=int(req_data.get("number_of_particles", 25)),
            box_size=float(req_data.get("box_size", 10.0)),
            cutoff=float(req_data.get("cutoff", 0.8)),
            spring_constant=float(req_data.get("spring_constant", 15.0)),
            equilibrium_distance=float(req_data.get("equilibrium_distance", 0.8)),
            initial_velocity_scale=float(req_data.get("initial_velocity_scale", 8.0)),
            minimum_initial_separation=float(req_data.get("minimum_initial_separation", 0.5)),
            dt=float(req_data.get("dt", 0.01)),
            steps=int(req_data.get("steps", 500)),
            gamma=float(req_data.get("gamma", 0.0)),
            integrator=str(req_data.get("integrator", "velocity_verlet")),
            dtype=str(req_data.get("dtype", "float64")),
            seed=int(req_data.get("seed", 0)),
            frame_stride=int(req_data.get("frame_stride", 1)),
        )
        config.validate()
        estimated_frames = config.steps // config.frame_stride + 1
        if config.number_of_particles > MAX_INTERACTIVE_PARTICLES:
            raise ValueError(f"number_of_particles must be at most {MAX_INTERACTIVE_PARTICLES}")
        if config.steps > MAX_INTERACTIVE_STEPS:
            raise ValueError(f"steps must be at most {MAX_INTERACTIVE_STEPS}")
        if estimated_frames > MAX_INTERACTIVE_FRAMES:
            raise ValueError(
                f"request would return {estimated_frames} frames; increase frame_stride "
                f"to return at most {MAX_INTERACTIVE_FRAMES} frames"
            )

        initial_state = create_random_state(config)
        init_diag = initial_state_diagnostics(initial_state, config.cutoff)
        result = run_simulation(config, initial_state)

        # Later frame diagnostics
        final_geom = pairwise_geometry(result.final_state.positions, config.cutoff)
        final_diag = {
            "final_interacting_pairs": int(np_count_pairs(final_geom.neighbors)),
            "final_mean_speed": float(result.mean_speed[-1]) if len(result.mean_speed) else 0.0,
        }

        # Convert arrays to python lists for JSON serialization
        response_payload = {
            "config": config.as_dict(),
            "max_relative_energy_drift": float(result.max_relative_energy_drift),
            "frame_times": result.frame_times.tolist(),
            "times": result.times.tolist(),
            "positions": result.positions.tolist(),  # shape: (num_frames, N, 3)
            "kinetic": result.kinetic.tolist(),
            "potential": result.potential.tolist(),
            "total_energy": result.total_energy.tolist(),
            "mean_speed": result.mean_speed.tolist(),
            "initial_diagnostics": init_diag,
            "final_diagnostics": final_diag,
        }
        return jsonify(response_payload)
    except Exception as err:
        return jsonify({"error": str(err)}), 400


def np_count_pairs(neighbors_matrix) -> int:
    import numpy as np
    return int(np.count_nonzero(np.triu(neighbors_matrix, k=1)))


@app.route("/api/results/<study_name>", methods=["GET"])
def get_study_results(study_name: str):
    """Read saved experiment CSV/JSON results from the results/ directory."""
    valid_studies = {
        "timestep": ROOT_DIR / "results" / "timestep" / "timestep_summary.csv",
        "dissipation": ROOT_DIR / "results" / "dissipation" / "dissipation_summary.csv",
        "precision": ROOT_DIR / "results" / "precision" / "precision_summary.csv",
        "memory": ROOT_DIR / "results" / "memory" / "pairwise_memory.csv",
        "performance": ROOT_DIR / "results" / "performance" / "benchmark.csv",
        "baseline": ROOT_DIR / "results" / "energy" / "baseline.json",
    }
    if study_name not in valid_studies:
        return jsonify({"error": f"Unknown study name '{study_name}'"}), 404

    target = valid_studies[study_name]
    if not target.exists():
        return jsonify({"exists": False, "message": f"No data found at {target.name}"})

    if target.suffix == ".csv":
        import csv
        with target.open(encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        return jsonify({"exists": True, "data": rows, "filename": target.name})
    elif target.suffix == ".json":
        data = json.loads(target.read_text(encoding="utf-8"))
        return jsonify({"exists": True, "data": data, "filename": target.name})

    return jsonify({"exists": False})


@app.route("/api/experiments/run", methods=["POST"])
def run_experiment_suite():
    """Run one experiment into a fresh output root, never into the frozen evidence.

    Because a browser click must not be able to overwrite the project's canonical
    results, this endpoint requires ``{"confirm": true}`` and writes to
    ``runs/<experiment>_<UTC timestamp>/``.
    """
    req = request.get_json(force=True) or {}
    experiment = req.get("experiment")
    if experiment not in VALID_EXPERIMENTS:
        return jsonify({"error": f"Unknown experiment '{experiment}'"}), 400
    if req.get("confirm") is not True:
        return jsonify({
            "status": "confirmation_required",
            "error": (
                "refusing to run: pass {\"experiment\": ..., \"confirm\": true}; "
                "results/ is frozen evidence and is never written by the dashboard"
            ),
        }), 428

    config_path = ROOT_DIR / "data" / "default_config.json"
    config = load_config(config_path) if config_path.exists() else SimulationConfig()

    try:
        output_dir = new_run_directory(experiment)
        if experiment == "timestep":
            path = timestep_sweep(config, output_dir / "timestep")
        elif experiment == "gamma":
            path = gamma_sweep(config, output_dir / "dissipation")
        elif experiment == "precision":
            path = precision_comparison(config, output_dir / "precision")
        elif experiment == "memory":
            path = memory_table(output_dir / "memory")
        else:
            path = save_benchmark_csv(
                benchmark_forces(), output_dir / "performance" / "benchmark.csv",
                meta=benchmark_protocol(),
            )
        return jsonify({
            "status": "success", "message": f"Saved to {path}", "output_root": str(output_dir),
        })
    except Exception as err:
        return jsonify({"error": str(err)}), 500


if __name__ == "__main__":
    print("Starting Atomic Toy Model Scientific Dashboard Server on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
