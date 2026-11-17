"""Kinetic and potential ."""

from __future__ import annotations

import numpy as np

from .geometry import pairwise_geometry


def kinetic_energy(velocities: np.ndarray, masses: np.ndarray) -> float:
    return float(0.5 * np.sum(masses * np.sum(velocities * velocities, axis=1)))


def potential_energy(
    positions: np.ndarray, cutoff: float, spring_constant: float, equilibrium_distance: float
) -> float:
    geometry = pairwise_geometry(positions, cutoff)
    upper_pairs = np.triu(geometry.neighbors, k=1)
    extension = geometry.distance[upper_pairs] - equilibrium_distance
    return float(0.5 * spring_constant * np.sum(extension * extension))


def total_energy(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    cutoff: float,
    spring_constant: float,
    equilibrium_distance: float,
) -> tuple[float, float, float]:
    kinetic = kinetic_energy(velocities, masses)
    potential = potential_energy(positions, cutoff, spring_constant, equilibrium_distance)
    return kinetic, potential, kinetic + potential
