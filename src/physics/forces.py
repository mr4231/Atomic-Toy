"""Harmonic pair-force implementations used for correctness and speed studies."""

from __future__ import annotations

import numpy as np

from .geometry import pairwise_geometry


def pair_forces_loop(
    positions: np.ndarray, cutoff: float, spring_constant: float, equilibrium_distance: float
) -> np.ndarray:
    """Compute spring forces with a transparent i<j reference double loop."""
    geometry = pairwise_geometry(positions, cutoff)
    forces = np.zeros_like(positions)
    n_particles = positions.shape[0]

    # Each unordered pair is visited once, then Newton's third law updates both particles.
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            if not geometry.neighbors[i, j]:
                continue
            distance = geometry.distance[i, j]
            direction = geometry.displacement[i, j] / distance
            force_on_i = -spring_constant * (distance - equilibrium_distance) * direction
            forces[i] += force_on_i
            forces[j] -= force_on_i
    return forces


def pair_forces_vectorized(
    positions: np.ndarray, cutoff: float, spring_constant: float, equilibrium_distance: float
) -> np.ndarray:
    """Compute the same harmonic forces with NumPy broadcasting for benchmarking."""
    geometry = pairwise_geometry(positions, cutoff)
    safe_distance = np.where(geometry.neighbors, geometry.distance, 1.0)
    unit_vectors = geometry.displacement / safe_distance[:, :, None]
    magnitude = -spring_constant * (geometry.distance - equilibrium_distance) * geometry.neighbors
    return np.sum(magnitude[:, :, None] * unit_vectors, axis=1)


def drag_forces(velocities: np.ndarray, gamma: float) -> np.ndarray:
    """Return viscous drag, which vanishes exactly for the conservative gamma=0 case."""
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    return -gamma * velocities


def total_forces(
    positions: np.ndarray,
    velocities: np.ndarray,
    cutoff: float,
    spring_constant: float,
    equilibrium_distance: float,
    gamma: float = 0.0,
    *,
    method: str = "vectorized",
) -> np.ndarray:
    """Combine conservative pair forces with optional velocity-proportional drag."""
    if method == "vectorized":
        pair = pair_forces_vectorized(positions, cutoff, spring_constant, equilibrium_distance)
    elif method == "loop":
        pair = pair_forces_loop(positions, cutoff, spring_constant, equilibrium_distance)
    else:
        raise ValueError("method must be 'vectorized' or 'loop'")
    return pair + drag_forces(velocities, gamma)
