
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.dissipation.analysis import relaxation_time
from src.model import SimulationConfig
from src.simulation.benchmarks import benchmark_forces, benchmark_protocol, save_benchmark_csv
from src.simulation.engine import create_random_state, run_simulation
from src.utils.evidence_guard import ProtectedEvidenceError, assert_safe_output
from src.utils.io import save_result
from src.utils.metrics import (
    normalized_position_difference,
    pairwise_storage_bytes,
    verification_metrics,
)


def load_config(path: str | Path) -> SimulationConfig:
    return SimulationConfig(**json.loads(Path(path).read_text(encoding="utf-8")))


def _write_csv(
    rows: list[dict[str, object]], path: Path, *, allow_protected_overwrite: bool = False
) -> Path:
    assert_safe_output(path, allow_protected_overwrite=allow_protected_overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_baseline(config: SimulationConfig, output_dir: str | Path) -> Path:
    initial = create_random_state(config)
    result = run_simulation(config, initial)
    output = Path(output_dir) / "baseline"
    save_result(result, output, initial_state=initial)
    initial_checks = verification_metrics(
        initial.positions, box_size=config.box_size, cutoff=config.cutoff,
        spring_constant=config.spring_constant, equilibrium_distance=config.equilibrium_distance,
    )
    trajectory_violation = max(
        verification_metrics(
            frame, box_size=config.box_size, cutoff=config.cutoff,
            spring_constant=config.spring_constant, equilibrium_distance=config.equilibrium_distance,
        )["max_box_violation"]
        for frame in result.positions
    )
    verification_path = Path(output_dir) / "verification_metrics.json"
    assert_safe_output(verification_path)
    verification_path.write_text(
        json.dumps(
            {
                "initial_state": initial_checks,
                "trajectory_max_box_violation": trajectory_violation,
                "frames_checked": len(result.positions),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def timestep_sweep(config: SimulationConfig, output_dir: str | Path, dts: tuple[float, ...] = (0.05, 0.02, 0.01, 0.005)) -> Path:
    rows: list[dict[str, object]] = []
    horizon = config.steps * config.dt
    initial = create_random_state(config)
    for dt in dts:
        steps = round(horizon / dt)
        if abs(steps * dt - horizon) > 1e-12:
            raise ValueError(f"dt={dt} does not divide the horizon {horizon} exactly")
        case = replace(config, dt=dt, steps=steps, gamma=0.0)
        result = run_simulation(case, initial)
        save_result(
            result, Path(output_dir) / f"dt_{dt:.4f}".replace(".", "p"), initial_state=initial
        )
        rows.append({
            "dt": dt, "steps": case.steps, "integrator": case.integrator,
            "max_relative_energy_drift": result.max_relative_energy_drift,
        })
    root = Path(output_dir)
    summary = _write_csv(rows, root / "timestep_summary.csv")
    from visualisation.plots import plot_energy, plot_energy_comparison

    stems = {}
    for dt in dts:
        label = f"dt={dt:g}"
        stem = root / f"dt_{dt:.4f}".replace(".", "p")
        stems[label] = stem
        plot_energy(stem, root / (f"energy_{dt:.4f}".replace(".", "p") + ".png"))
    plot_energy_comparison(stems, root / "total_energy_by_timestep.png", title="Total energy by timestep")
    return summary


def gamma_sweep(config: SimulationConfig, output_dir: str | Path, gammas: tuple[float, ...] = (0.0, 0.05, 0.1, 0.5)) -> Path:
    initial = create_random_state(config)
    rows: list[dict[str, object]] = []
    for gamma in gammas:
        case = replace(config, gamma=gamma)
        result = run_simulation(case, initial)
        save_result(
            result, Path(output_dir) / f"gamma_{gamma:.3f}".replace(".", "p"), initial_state=initial
        )
        settled = relaxation_time(result.times, result.total_energy, result.mean_speed)
        rows.append({
            "gamma": gamma, "steps": case.steps, "final_total_energy": float(result.total_energy[-1]),
            "final_mean_speed": float(result.mean_speed[-1]),
            "relaxation_time": "not_reached" if settled is None else settled,
            "criterion": "relative_energy_change<3e-3 AND mean_speed<3e-2 over 20 steps",
        })
    root = Path(output_dir)
    summary = _write_csv(rows, root / "dissipation_summary.csv")
    from visualisation.plots import plot_energy, plot_energy_comparison

    stems = {}
    for gamma in gammas:
        label = f"gamma={gamma:g}"
        stem = root / f"gamma_{gamma:.3f}".replace(".", "p")
        stems[label] = stem
        plot_energy(stem, root / (f"energy_{gamma:.3f}".replace(".", "p") + ".png"))
    plot_energy_comparison(stems, root / "total_energy_by_gamma.png", title="Conservative and dissipative energy")
    return summary


def precision_comparison(
    config: SimulationConfig, output_dir: str | Path, *, timing_repeats: int = 5
) -> Path:
    if timing_repeats < 5:
        raise ValueError("precision timing requires at least five repeats")
    source_state = create_random_state(replace(config, dtype="float64"))
    results = {}
    cases = {}
    for dtype in ("float32", "float64"):
        case = replace(config, dtype=dtype)
        cases[dtype] = case
        state = type(source_state)(
            source_state.positions.astype(dtype), source_state.velocities.astype(dtype),
            source_state.masses.astype(dtype), source_state.step,
        )
        results[dtype] = run_simulation(case, state)
        save_result(results[dtype], Path(output_dir) / dtype, initial_state=state)
    rows = []
    reference = results["float64"]
    for dtype, result in results.items():
        rows.append({
            "dtype": dtype,
            "position_bytes": result.final_state.positions.nbytes,
            "pair_displacement_bytes": pairwise_storage_bytes(config.n_particles, dtype),
            "max_relative_energy_drift": result.max_relative_energy_drift,
            "final_position_difference_vs_float64": normalized_position_difference(
                reference.final_state.positions, result.final_state.positions.astype("float64")
            ),
            "execution_median_seconds": float(statistics.median(
                _timed_run(cases[dtype], source_state, dtype) for _ in range(timing_repeats)
            )),
            "timing_repeats": timing_repeats,
            "memory_ratio_vs_float32": result.final_state.positions.nbytes /
            results["float32"].final_state.positions.nbytes,
        })
    root = Path(output_dir)
    summary = _write_csv(rows, root / "precision_summary.csv")
    from visualisation.plots import plot_precision_runtime

    plot_precision_runtime(summary, root / "precision_runtime.png")
    return summary


def _timed_run(config: SimulationConfig, source_state, dtype: str) -> float:
    state = type(source_state)(
        source_state.positions.astype(dtype), source_state.velocities.astype(dtype),
        source_state.masses.astype(dtype), source_state.step,
    )
    started = time.perf_counter()
    run_simulation(config, state)
    return time.perf_counter() - started


def memory_table(output_dir: str | Path) -> Path:
    rows = [
        {"n_particles": n, "dtype": dtype, "pair_displacement_bytes": pairwise_storage_bytes(n, dtype),
         "pair_displacement_mebibytes": pairwise_storage_bytes(n, dtype) / 1024**2}
        for n in (100, 1000, 5000, 10000) for dtype in ("float32", "float64")
    ]
    return _write_csv(rows, Path(output_dir) / "pairwise_memory.csv")


def performance_study(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    summary = save_benchmark_csv(
        benchmark_forces(), root / "benchmark.csv", meta=benchmark_protocol()
    )
    from visualisation.plots import plot_performance

    plot_performance(summary, root / "performance.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("baseline", "timestep", "gamma", "precision", "memory", "performance", "all"))
    parser.add_argument("--config", default="data/default_config.json")
    parser.add_argument("--output-root", default="results")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(args.output_root)
    actions = {
        "baseline": lambda: run_baseline(config, root / "energy"),
        "timestep": lambda: timestep_sweep(config, root / "timestep"),
        "gamma": lambda: gamma_sweep(config, root / "dissipation"),
        "precision": lambda: precision_comparison(config, root / "precision"),
        "memory": lambda: memory_table(root / "memory"),
        "performance": lambda: performance_study(root / "performance"),
    }
    selected = actions if args.experiment == "all" else {args.experiment: actions[args.experiment]}
    try:
        for name, action in selected.items():
            print(f"{name}: {action()}")
    except ProtectedEvidenceError as error:
        # A refusal must be an explicit, readable message, never a silent overwrite.
        raise SystemExit(f"refused: {error}") from None


if __name__ == "__main__":
    main()
