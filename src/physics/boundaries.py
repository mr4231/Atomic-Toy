"""Hard-wall reflection for a cubic 3D simulation box."""

from __future__ import annotations

import numpy as np


def reflect_walls(positions: np.ndarray, velocities: np.ndarray, box_length: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact specular reflection against the hard walls, folded for multi-crossing.

    For each component the number of wall crossings in the last step is
    ``floor(x / L)`` (negative for the lower wall).  An odd crossing count means
    the particle ends a crossing *outward*, so the normal velocity flips and the
    position mirrors back into the box (``x -> 2L - x`` after folding).  This is
    the exact, time-reversible specular rule (D-06); speed magnitude is preserved
    and only the crossed normal component changes.
    """
    if box_length <= 0:
        raise ValueError("box_length must be positive")
    if positions.shape != velocities.shape or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions and velocities must both have shape (N, 3)")

    for axis in range(3):
        component = positions[:, axis]
        crossings = np.floor(component / box_length).astype(np.int64)
        folded = component - crossings * box_length
        odd = (crossings % 2) != 0
        positions[:, axis] = np.where(odd, box_length - folded, folded)
        velocities[odd, axis] *= -1.0
    return positions, velocities
