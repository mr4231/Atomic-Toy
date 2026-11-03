"""Kinetic and harmonic-pair potential energy accounting."""

from __future__ import annotations

import numpy as np

from .geometry import pairwise_geometry


def kinetic_energy(velocities: np.ndarray, masses: np.ndarray) -> float:
    """Return 1/2 sum_i m_i |v_i|^2."""
    return float(0.5 * np.sum(masses * np.sum(velocities * velocities, axis=1)))


def potential_energy(
    positions: np.ndarray, cutoff: float, spring_constant: float, equilibrium_distance: float
) -> float:
    """Sum harmonic pair potentials once over i<j, avoiding double counting."""
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
    """Return kinetic, potential, and mechanical total energy in that order."""
    kinetic = kinetic_energy(velocities, masses)
    potential = potential_energy(positions, cutoff, spring_constant, equilibrium_distance)
    return kinetic, potential, kinetic + potential
