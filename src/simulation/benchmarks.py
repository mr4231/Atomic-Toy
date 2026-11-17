"""Controlled Python, NumPy, and optional C pair-force benchmarks."""

from __future__ import annotations

import ctypes
import csv
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np

from src.physics.cell_list import pair_forces_cell_list
from src.physics.forces import pair_forces_loop, pair_forces_vectorized
from src.utils.evidence_guard import assert_safe_output
from src.utils.provenance import PROVENANCE_KEY, provenance_record

# Protocol defaults, kept in one place so a saved timing file can record exactly
# which protocol produced it (the values were previously only in the signature).
DEFAULT_PARTICLE_COUNTS = (50, 200, 500, 1000)
DEFAULT_REPEATS = 5
DEFAULT_CUTOFF = 2.5
DEFAULT_SPRING_CONSTANT = 15.0
DEFAULT_EQUILIBRIUM_DISTANCE = 0.8
DEFAULT_SEED = 123
DEFAULT_BOX_SIZE = 10.0
# No discarded warm-up calls are made today; recorded so the gap is visible.
DEFAULT_WARMUP = 0


def benchmark_protocol() -> dict[str, object]:
    """Return the protocol the default ``benchmark_forces`` call uses."""
    return {
        "particle_counts": list(DEFAULT_PARTICLE_COUNTS),
        "repeats": DEFAULT_REPEATS,
        "warmup": DEFAULT_WARMUP,
        "cutoff": DEFAULT_CUTOFF,
        "spring_constant": DEFAULT_SPRING_CONSTANT,
        "equilibrium_distance": DEFAULT_EQUILIBRIUM_DISTANCE,
        "seed": DEFAULT_SEED,
        "box_size": DEFAULT_BOX_SIZE,
    }


def load_c_force() -> Callable[[np.ndarray, float, float, float], np.ndarray]:
    from c_backend.build import build

    library_path = build("-O2")
    library = ctypes.CDLL(str(library_path))
    function = library.pair_forces_3d
    function.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    function.restype = None

    def c_force(positions: np.ndarray, cutoff: float, spring_constant: float, equilibrium_distance: float) -> np.ndarray:
        contiguous = np.ascontiguousarray(positions, dtype=np.float64)
        output = np.zeros_like(contiguous)
        function(
            contiguous.shape[0], contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), cutoff,
            spring_constant, equilibrium_distance,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return output

    return c_force


def benchmark_forces(
    particle_counts: tuple[int, ...] = DEFAULT_PARTICLE_COUNTS, repeats: int = DEFAULT_REPEATS,
    cutoff: float = DEFAULT_CUTOFF, spring_constant: float = DEFAULT_SPRING_CONSTANT,
    equilibrium_distance: float = DEFAULT_EQUILIBRIUM_DISTANCE,
    seed: int = DEFAULT_SEED, box_size: float = DEFAULT_BOX_SIZE,
) -> list[dict[str, float | int | str]]:
    if repeats < 5:
        raise ValueError("the project timing protocol requires at least five repeats")
    
    def cell_list_wrapper(pos, cut, k, r0):
        return pair_forces_cell_list(pos, box_size, cut, k, r0)

    implementations: dict[str, Callable[[np.ndarray, float, float, float], np.ndarray]] = {
        "python_loop": pair_forces_loop,
        "numpy_vectorized": pair_forces_vectorized,
        "cell_list_spatial": cell_list_wrapper,
    }
    try:
        implementations["c_optimized"] = load_c_force()
    except (OSError, RuntimeError, subprocess.CalledProcessError, ImportError, AttributeError, NameError):
        pass


    rows: list[dict[str, float | int | str]] = []
    for n_particles in particle_counts:
        positions = np.random.default_rng(seed + n_particles).uniform(0.0, box_size, (n_particles, 3))
        reference = pair_forces_loop(positions, cutoff, spring_constant, equilibrium_distance)
        reference_time: float | None = None
        for name, implementation in implementations.items():
            candidate = implementation(positions, cutoff, spring_constant, equilibrium_distance)
            parity_error = float(np.linalg.norm(reference - candidate))
            if not np.allclose(reference, candidate, rtol=1e-10, atol=1e-10):
                raise AssertionError(f"{name} disagrees with the reference: norm={parity_error}")
            samples = []
            for _ in range(repeats):
                start = time.perf_counter()
                implementation(positions, cutoff, spring_constant, equilibrium_distance)
                samples.append(time.perf_counter() - start)
            median = float(statistics.median(samples))
            if name == "python_loop":
                reference_time = median
            rows.append({
                "implementation": name,
                "n_particles": n_particles,
                "repeats": repeats,
                "median_seconds": median,
                "speedup_vs_python": float(reference_time / median) if reference_time else 1.0,
                "force_parity_norm": parity_error,
            })
    return rows


def save_benchmark_csv(
    rows: list[dict[str, float | int | str]], path: str | Path, *, meta: dict[str, object] | None = None,
    allow_protected_overwrite: bool = False,
) -> Path:
    output = Path(path)
    sidecar = output.with_suffix(".meta.json")
    assert_safe_output(output, allow_protected_overwrite=allow_protected_overwrite)
    assert_safe_output(sidecar, allow_protected_overwrite=allow_protected_overwrite)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "csv": output.name,
        "rows": len(rows),
        "implementations": sorted({str(row["implementation"]) for row in rows}),
        "particle_counts": sorted({int(row["n_particles"]) for row in rows}),
        "protocol": benchmark_protocol() if meta is None else meta,
        PROVENANCE_KEY: provenance_record(
            extra={
                "timing_reproducible": False,
                "note": "wall-clock columns are machine dependent; parity norms and inputs are reproducible",
            }
        ),
    }
    output.with_suffix(".meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output
