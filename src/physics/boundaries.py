"""Wall reflection ."""

from __future__ import annotations

import numpy as np


def reflect_walls(positions: np.ndarray, velocities: np.ndarray, box_length: float) -> tuple[np.ndarray, np.ndarray]:

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
