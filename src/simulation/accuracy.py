"""Gate-G3 accuracy-study machinery (D-10): timestep convergence against a high-resolution reference.

The design is frozen before any run (see ``DECISION_LOG.md`` D-10 and
``docs/EXPERIMENT_PROTOCOL.md``).  This module implements the protocol
mechanics only; it deliberately does **not** choose the final timestep or
integrator and has not been run at the official settings yet.

Protocol implemented here:

- timestep set ``dt = (0.05, 0.02, 0.01, 0.005)`` for both ``semi_implicit_euler``
  and ``velocity_verlet`` at the common physical horizon ``steps * dt`` of the
  base configuration (``data/configs/accuracy_interacting.json``);
- one shared initial condition for every case: the state is created once (or
  passed explicitly) and ``run_simulation`` copies it per run, so no case can
  drift onto a different IC;
- a high-resolution reference trajectory on the same IC (velocity Verlet at
  ``dt = 0.000625`` by default).  The reference is explicitly *not* exact; a
  reference-refinement run (``dt = 0.0003125``) quantifies how much the
  reference itself moves under refinement;
- common physical sample times: multiples of a grid interval that **every**
  study dt divides exactly — by default the least common multiple of the dts
  (0.1 for the official set, since e.g. 0.05 is not a multiple of 0.02) —
  through the horizon.  Recorded frame times are ``step * dt``, so alignment
  is exact by construction and enforced numerically;
- per-case metrics: ``max_relative_energy_drift`` (from ``SimulationResult``);
  ``position_error`` = max over common sample times of
  ``||x_case(t) - x_ref(t)||_2 / (box_size * sqrt(N))``; and
  ``final_velocity_error`` = ``||v_case(T) - v_ref(T)||_2 / (v_scale * sqrt(N))``
  at the common horizon ``T``.  Per-frame velocities are not part of the stored
  NPZ contract, so the velocity error is measured at the final time only (the
  definitions are recorded in ``metric_definitions`` in the manifest);
- measured pairwise convergence order ``p = log(e_coarse/e_fine)/log(dt_coarse/dt_fine)``
  is recorded as measured.  No second-order behaviour is claimed by this module.

Every artifact is written to a NEW output root; writes are refused by
``src/utils/evidence_guard.py`` when they would touch frozen study directories
or existing files under ``results/``.  Nothing here changes the force law,
potential, wall behaviour, drag, precision defaults or public APIs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.model import SimulationConfig, SimulationState
from src.simulation.engine import create_random_state, run_simulation
from src.utils.evidence_guard import ProtectedEvidenceError, assert_safe_output
from src.utils.io import save_result
from src.utils.provenance import (
    config_fingerprint,
    initial_condition_fingerprint,
    provenance_record,
)

DEFAULT_DTS: tuple[float, ...] = (0.05, 0.02, 0.01, 0.005)
DEFAULT_INTEGRATORS: tuple[str, ...] = ("semi_implicit_euler", "velocity_verlet")
DEFAULT_REFERENCE_DT = 0.000625
DEFAULT_REFINEMENT_DT = 0.0003125

SHORT_INTEGRATOR = {
    "semi_implicit_euler": "euler",
    "velocity_verlet": "verlet",
}


def _least_common_interval(dts: Iterable[float]) -> float:
    """Smallest interval of which every dt is an exact integer multiple.

    Computed in exact rational arithmetic so float representation cannot pick a
    grid that some case cannot hit: 0.05 is a multiple of 0.05 and 0.01 but NOT
    of 0.02, so the official set shares the 0.1 grid, not the coarsest dt.
    """
    fractions = [Fraction(dt).limit_denominator(10**9) for dt in dts]
    common = Fraction(
        math.lcm(*(f.numerator for f in fractions)),
        math.gcd(*(f.denominator for f in fractions)),
    )
    return float(common)


def _steps_for_horizon(horizon: float, dt: float) -> int:
    """Number of ``dt`` steps that reach ``horizon`` exactly (no silent rounding)."""
    steps = round(horizon / dt)
    if steps < 0 or abs(steps * dt - horizon) > 1e-9:
        raise ValueError(f"dt={dt} does not divide the horizon {horizon} exactly")
    return steps


def _steps_per_sample(sample_interval: float, dt: float) -> int:
    """Steps of ``dt`` between common sample times (``sample_interval`` must be a multiple)."""
    per_sample = round(sample_interval / dt)
    if per_sample < 1 or abs(per_sample * dt - sample_interval) > 1e-9:
        raise ValueError(
            f"sample_interval={sample_interval} is not an exact multiple of dt={dt}"
        )
    return per_sample


def sample_times(horizon: float, sample_interval: float) -> np.ndarray:
    """Common physical sample times: ``0, sample_interval, ..., horizon`` inclusive."""
    count = _steps_for_horizon(horizon, sample_interval)
    return np.arange(count + 1, dtype=float) * sample_interval


def frame_indices_for_times(
    frame_times: np.ndarray, wanted_times: Iterable[float], tol: float = 1e-9
) -> np.ndarray:
    """Indices of ``frame_times`` matching each wanted time exactly (within ``tol``)."""
    frame_times = np.asarray(frame_times, dtype=float)
    indices = []
    for wanted in np.asarray(list(wanted_times), dtype=float):
        distance = np.abs(frame_times - wanted)
        index = int(np.argmin(distance))
        if distance[index] > tol:
            raise ValueError(
                f"sample time {wanted} is not present in frame times "
                f"(nearest recorded {frame_times[index]})"
            )
        indices.append(index)
    return np.asarray(indices, dtype=int)


def aligned_samples(
    result: Any, wanted_times: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Recorded positions and their times at the common sample times."""
    indices = frame_indices_for_times(result.frame_times, wanted_times)
    return result.positions[indices], result.frame_times[indices]


def normalized_error(candidate: np.ndarray, reference: np.ndarray, scale: float) -> float:
    """``||candidate - reference||_2 / scale`` with the Frobenius norm over all components."""
    difference = np.asarray(candidate, dtype=float) - np.asarray(reference, dtype=float)
    return float(np.linalg.norm(difference.ravel()) / scale)


def trajectory_position_error(
    case_result: Any, reference_result: Any, wanted_times: Iterable[float], scale: float
) -> float:
    """Max over common sample times of the normalized position error against the reference."""
    case_positions, case_times = aligned_samples(case_result, wanted_times)
    reference_positions, reference_times = aligned_samples(reference_result, wanted_times)
    if not np.allclose(case_times, reference_times, atol=1e-9):
        raise ValueError("case and reference sample times disagree")
    errors = [
        normalized_error(case, reference, scale)
        for case, reference in zip(case_positions, reference_positions)
    ]
    return float(max(errors))


def _convergence_order(
    error_coarse: float, error_fine: float, dt_coarse: float, dt_fine: float
) -> float | None:
    """``p = log(e_coarse/e_fine)/log(dt_coarse/dt_fine)``; ``None`` when not measurable."""
    if dt_coarse <= 0 or dt_fine <= 0 or dt_coarse == dt_fine:
        return None
    if error_coarse <= 0 or error_fine <= 0:
        return None  # a zero or negative error leaves no measurable order
    order = math.log(error_coarse / error_fine) / math.log(dt_coarse / dt_fine)
    return order if math.isfinite(order) else None


def measured_orders(
    dts: Iterable[float], errors: Iterable[float], integrator: str
) -> list[dict[str, Any]]:
    """Adjacent-pair and overall convergence orders, sorted by descending dt.

    The values are recorded exactly as measured; no convergence behaviour is
    claimed by this module (DECISION_LOG.md D-10).
    """
    pairs = sorted(zip(tuple(dts), tuple(errors)), key=lambda item: item[0], reverse=True)
    rows: list[dict[str, Any]] = []
    for (dt_coarse, error_coarse), (dt_fine, error_fine) in zip(pairs, pairs[1:]):
        rows.append(
            {
                "comparison": "adjacent",
                "integrator": integrator,
                "dt_coarse": dt_coarse,
                "dt_fine": dt_fine,
                "error_coarse": error_coarse,
                "error_fine": error_fine,
                "order": _convergence_order(error_coarse, error_fine, dt_coarse, dt_fine),
            }
        )
    if len(pairs) >= 2:
        (dt_coarse, error_coarse), (dt_fine, error_fine) = pairs[0], pairs[-1]
        rows.append(
            {
                "comparison": "overall",
                "integrator": integrator,
                "dt_coarse": dt_coarse,
                "dt_fine": dt_fine,
                "error_coarse": error_coarse,
                "error_fine": error_fine,
                "order": _convergence_order(error_coarse, error_fine, dt_coarse, dt_fine),
            }
        )
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    """Write a summary CSV, refusing to touch frozen or already-written evidence."""
    assert_safe_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    assert_safe_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def run_accuracy_study(
    config: SimulationConfig,
    output_root: str | Path,
    *,
    dts: Iterable[float] = DEFAULT_DTS,
    integrators: Iterable[str] = DEFAULT_INTEGRATORS,
    reference_dt: float = DEFAULT_REFERENCE_DT,
    refinement_dt: float = DEFAULT_REFINEMENT_DT,
    sample_interval: float | None = None,
    initial_state: SimulationState | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    """Run the controlled accuracy study and write every artifact to a NEW output root.

    All cases share one initial condition; the reference and its refinement use
    the same IC with velocity Verlet at substantially smaller timesteps.  The
    reference is not claimed to be exact; the manifest records how much it moves
    under refinement.  Returns the manifest dictionary that was written.
    """
    config.validate()
    if config.gamma != 0.0:
        raise ValueError(
            "the accuracy study requires gamma = 0 (convergence and energy-drift "
            f"metrics assume conservative dynamics); got gamma={config.gamma}"
        )
    dts = tuple(float(dt) for dt in dts)
    integrators = tuple(integrators)
    if not dts or not integrators:
        raise ValueError("at least one dt and one integrator are required")
    for integrator in integrators:
        if integrator not in SHORT_INTEGRATOR:
            raise ValueError(f"unknown integrator {integrator!r}")

    horizon = config.steps * config.dt
    if sample_interval is None:
        sample_interval = _least_common_interval(dts)
    for dt in (*dts, reference_dt, refinement_dt):
        _steps_for_horizon(horizon, dt)
        _steps_per_sample(sample_interval, dt)
    wanted_times = sample_times(horizon, sample_interval)

    # One shared initial condition: run_simulation copies the state per run.
    initial = initial_state if initial_state is not None else create_random_state(config)
    ic_sha256 = initial_condition_fingerprint(initial)
    position_scale = config.box_size * math.sqrt(config.n_particles)
    velocity_scale = (
        config.initial_velocity_scale if config.initial_velocity_scale > 0 else 1.0
    ) * math.sqrt(config.n_particles)

    root = Path(output_root)

    reference_config = replace(
        config,
        dt=reference_dt,
        steps=_steps_for_horizon(horizon, reference_dt),
        integrator="velocity_verlet",
        frame_stride=_steps_per_sample(sample_interval, reference_dt),
    )
    reference_result = run_simulation(reference_config, initial)
    save_result(reference_result, root / "reference", initial_state=initial, command=command)
    reference_sha = config_fingerprint(reference_config)
    reference_entry = {
        "output_stem": str(root / "reference"),
        "config_sha256": reference_sha,
        "dt": reference_dt,
        "steps": reference_config.steps,
        "frame_stride": reference_config.frame_stride,
        "integrator": "velocity_verlet",
    }

    refinement_config = replace(
        config,
        dt=refinement_dt,
        steps=_steps_for_horizon(horizon, refinement_dt),
        integrator="velocity_verlet",
        frame_stride=_steps_per_sample(sample_interval, refinement_dt),
    )
    refinement_result = run_simulation(refinement_config, initial)
    save_result(
        refinement_result, root / "reference_refinement", initial_state=initial, command=command
    )
    refinement_sha = config_fingerprint(refinement_config)
    refinement_entry = {
        "output_stem": str(root / "reference_refinement"),
        "config_sha256": refinement_sha,
        "dt": refinement_dt,
        "steps": refinement_config.steps,
        "frame_stride": refinement_config.frame_stride,
        "integrator": "velocity_verlet",
    }

    reference_self_consistency = {
        "position_error": trajectory_position_error(
            refinement_result, reference_result, wanted_times, position_scale
        ),
        "final_velocity_error": normalized_error(
            refinement_result.final_state.velocities,
            reference_result.final_state.velocities,
            velocity_scale,
        ),
    }

    csv_rows: list[dict[str, Any]] = []
    case_entries: list[dict[str, Any]] = []
    errors_by_integrator: dict[str, dict[float, float]] = {}
    for integrator in integrators:
        errors_by_integrator[integrator] = {}
        for dt in dts:
            case = replace(
                config,
                dt=dt,
                steps=_steps_for_horizon(horizon, dt),
                integrator=integrator,
                frame_stride=1,
            )
            result = run_simulation(case, initial)
            position_error = trajectory_position_error(
                result, reference_result, wanted_times, position_scale
            )
            velocity_error = normalized_error(
                result.final_state.velocities,
                reference_result.final_state.velocities,
                velocity_scale,
            )
            stem = root / (f"dt_{dt:.4f}".replace(".", "p") + f"_{SHORT_INTEGRATOR[integrator]}")
            save_result(result, stem, initial_state=initial, command=command)
            row = {
                "dt": dt,
                "steps": case.steps,
                "integrator": integrator,
                "n_particles": config.n_particles,
                "box_size": config.box_size,
                "cutoff": config.cutoff,
                "spring_constant": config.spring_constant,
                "equilibrium_distance": config.equilibrium_distance,
                "initial_velocity_scale": config.initial_velocity_scale,
                "minimum_initial_separation": config.minimum_initial_separation,
                "gamma": config.gamma,
                "dtype": config.dtype,
                "seed": config.seed,
                "frame_stride": case.frame_stride,
                "t_end": horizon,
                "sample_interval": sample_interval,
                "reference_dt": reference_dt,
                "ic_sha256": ic_sha256,
                "reference_config_sha256": reference_sha,
                "max_relative_energy_drift": result.max_relative_energy_drift,
                "position_error": position_error,
                "final_velocity_error": velocity_error,
            }
            csv_rows.append(row)
            case_entries.append({**row, "output_stem": str(stem)})
            errors_by_integrator[integrator][dt] = position_error

    order_rows: list[dict[str, Any]] = []
    ratios: dict[str, float | None] = {}
    for integrator in integrators:
        integrator_errors = errors_by_integrator[integrator]
        order_rows.extend(
            measured_orders(
                list(integrator_errors.keys()), list(integrator_errors.values()), integrator
            )
        )
        finest_dt = min(integrator_errors)
        finest_error = integrator_errors[finest_dt]
        ratios[integrator] = (
            reference_self_consistency["position_error"] / finest_error
            if finest_error > 0
            else None
        )

    manifest: dict[str, Any] = {
        "experiment": "accuracy_study",
        "metric_definitions": {
            "position_error": (
                "max over common sample times of ||x_case(t) - x_reference(t)||_2 / "
                "(box_size * sqrt(N))"
            ),
            "final_velocity_error": (
                "||v_case(T) - v_reference(T)||_2 / (initial_velocity_scale * sqrt(N)) "
                "at the common horizon T"
            ),
            "max_relative_energy_drift": (
                "from SimulationResult: max |E(t) - E(0)| / max(|E(0)|, eps)"
            ),
            "order": (
                "p = log(e_coarse/e_fine) / log(dt_coarse/dt_fine), measured pairwise; "
                "null when not measurable; no convergence behaviour is claimed here"
            ),
        },
        "horizon": horizon,
        "dts": list(dts),
        "integrators": list(integrators),
        "reference_dt": reference_dt,
        "refinement_dt": refinement_dt,
        "sample_interval": sample_interval,
        "n_sample_times": int(len(wanted_times)),
        "reference_config": reference_config.as_dict(),
        "refinement_config": refinement_config.as_dict(),
        "reference_config_sha256": reference_sha,
        "refinement_config_sha256": refinement_sha,
        "initial_condition_sha256": ic_sha256,
        "reference": reference_entry,
        "reference_refinement": refinement_entry,
        "reference_self_consistency": reference_self_consistency,
        "reference_refinement_to_finest_ratio": ratios,
        "cases": case_entries,
        "orders": order_rows,
        "provenance": provenance_record(
            initial_state=initial,
            config=config,
            command=command,
            extra={"output_root": str(root)},
        ),
    }
    _write_csv(csv_rows, root / "accuracy_summary.csv")
    _write_csv(order_rows, root / "convergence_orders.csv")
    _write_json(manifest, root / "accuracy_manifest.json")
    return manifest


def main() -> None:
    """CLI entry point; the run itself is executed separately into a new dated root."""
    parser = argparse.ArgumentParser(description="Gate-G3 accuracy study (design frozen in D-10)")
    parser.add_argument("--config", default="data/configs/accuracy_interacting.json")
    parser.add_argument("--output-root", default="results/accuracy_2026-09-22")
    parser.add_argument("--dts", default=",".join(str(dt) for dt in DEFAULT_DTS))
    parser.add_argument("--integrators", default=",".join(DEFAULT_INTEGRATORS))
    parser.add_argument("--reference-dt", type=float, default=DEFAULT_REFERENCE_DT)
    parser.add_argument("--refinement-dt", type=float, default=DEFAULT_REFINEMENT_DT)
    parser.add_argument("--sample-interval", type=float, default=None)
    args = parser.parse_args()
    config = SimulationConfig(**json.loads(Path(args.config).read_text(encoding="utf-8")))
    dts = tuple(float(value) for value in args.dts.split(",") if value.strip())
    integrators = tuple(value.strip() for value in args.integrators.split(",") if value.strip())
    try:
        manifest = run_accuracy_study(
            config,
            args.output_root,
            dts=dts,
            integrators=integrators,
            reference_dt=args.reference_dt,
            refinement_dt=args.refinement_dt,
            sample_interval=args.sample_interval,
        )
    except ProtectedEvidenceError as error:
        # A refusal must be an explicit, readable message, never a silent overwrite.
        raise SystemExit(f"refused: {error}") from None
    for row in manifest["cases"]:
        print(
            f"dt={row['dt']:<7} {row['integrator']:<19} "
            f"drift={row['max_relative_energy_drift']:.3e} "
            f"pos_err={row['position_error']:.3e} "
            f"vel_err={row['final_velocity_error']:.3e}"
        )
    print(f"manifest: {Path(args.output_root) / 'accuracy_manifest.json'}")


if __name__ == "__main__":
    main()
