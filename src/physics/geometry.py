"""cutoff-neighbor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairGeometry:
    displacement: np.ndarray
    distance: np.ndarray
    neighbors: np.ndarray


def pairwise_geometry(positions: np.ndarray, cutoff: float, *, epsilon: float = 1e-12) -> PairGeometry:
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")

    # Broadcasting creates all r_i - r_j vectors without Python loops.
    displacement = positions[:, None, :] - positions[None, :, :]
    distance = np.linalg.norm(displacement, axis=2)
    neighbors = (distance > epsilon) & (distance <= cutoff)
    np.fill_diagonal(neighbors, False)
    return PairGeometry(displacement=displacement, distance=distance, neighbors=neighbors)
