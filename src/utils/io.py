"""Portable NPZ/JSON result storage for plots, notebooks, and reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.model import SimulationConfig, SimulationResult, SimulationState
from src.utils.evidence_guard import assert_safe_output
from src.utils.provenance import PROVENANCE_KEY, provenance_record


def save_result(
    result: SimulationResult,
    output_stem: str | Path,
    *,
    initial_state: SimulationState | None = None,
    command: str | None = None,
    allow_protected_overwrite: bool = False,
) -> tuple[Path, Path]:
    """Save numeric arrays to NPZ and configuration/summary metadata to JSON.

    ``initial_state`` is optional and only used to fingerprint the run's initial
    conditions; pass it whenever it is available so two runs can be compared.
    The write is refused when it would touch frozen evidence (see
    ``src/utils/evidence_guard.py``); pass ``allow_protected_overwrite=True`` only
    when replacing a canonical file is deliberate.
    """
    stem = Path(output_stem)
    npz_path = stem.with_suffix(".npz")
    json_path = stem.with_suffix(".json")
    # Fail before anything is written, so a refusal never leaves a partial result.
    assert_safe_output(npz_path, allow_protected_overwrite=allow_protected_overwrite)
    assert_safe_output(json_path, allow_protected_overwrite=allow_protected_overwrite)
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path, times=result.times, frame_times=result.frame_times, positions=result.positions, kinetic=result.kinetic,
        potential=result.potential, total_energy=result.total_energy, mean_speed=result.mean_speed,
        final_positions=result.final_state.positions, final_velocities=result.final_state.velocities,
        masses=result.final_state.masses,
    )
    metadata = {
        "config": result.config.as_dict(),
        "max_relative_energy_drift": result.max_relative_energy_drift,
        "frames": int(result.positions.shape[0]),
        PROVENANCE_KEY: provenance_record(
            initial_state=initial_state, config=result.config, command=command
        ),
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return npz_path, json_path


def load_result(output_stem: str | Path) -> SimulationResult:
    """Reload a saved run so visualization does not rerun or alter simulation data."""
    stem = Path(output_stem)
    arrays = np.load(stem.with_suffix(".npz"))
    metadata = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    config = SimulationConfig(**metadata["config"])
    final_state = SimulationState(
        arrays["final_positions"], arrays["final_velocities"], arrays["masses"], config.steps
    )
    return SimulationResult(
        arrays["times"], arrays["frame_times"], arrays["positions"], arrays["kinetic"], arrays["potential"],
        arrays["total_energy"], arrays["mean_speed"], config, final_state,
    )
