"""Metrics used in precision, memory, and verification studies."""

from __future__ import annotations

import numpy as np

from src.physics.forces import pair_forces_vectorized
from src.physics.geometry import pairwise_geometry


def pairwise_storage_bytes(n_particles: int, dtype: str | np.dtype, dimensions: int = 3) -> int:
    """Return dense (N,N,dimensions) displacement-array storage without allocating it."""
    if n_particles < 0 or dimensions < 1:
        raise ValueError("n_particles and dimensions must be non-negative/positive")
    return n_particles * n_particles * dimensions * np.dtype(dtype).itemsize


def normalized_position_difference(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Scale final-state position error by the reference norm for dtype studies."""
    if reference.shape != candidate.shape:
        raise ValueError("arrays must have matching shapes")
    denominator = max(float(np.linalg.norm(reference)), np.finfo(reference.dtype).eps)
    return float(np.linalg.norm(reference - candidate) / denominator)


def verification_metrics(
    positions: np.ndarray,
    *,
    box_size: float,
    cutoff: float,
    spring_constant: float,
    equilibrium_distance: float,
) -> dict[str, float | int]:
    """Return report-ready numerical checks for one particle configuration."""
    geometry = pairwise_geometry(positions, cutoff)
    forces = pair_forces_vectorized(
        positions, cutoff, spring_constant, equilibrium_distance
    )
    lower_violation = np.maximum(0.0, -positions)
    upper_violation = np.maximum(0.0, positions - box_size)
    return {
        "distance_symmetry_max_abs": float(np.max(np.abs(geometry.distance - geometry.distance.T))),
        "distance_diagonal_max_abs": float(np.max(np.abs(np.diag(geometry.distance)))),
        "self_neighbor_count": int(np.count_nonzero(np.diag(geometry.neighbors))),
        "net_internal_force_norm": float(np.linalg.norm(np.sum(forces, axis=0))),
        "max_box_violation": float(max(np.max(lower_violation), np.max(upper_violation))),
    }
