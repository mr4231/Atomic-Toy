"""3D Cell-List (Linked-Cell) Spatial Partitioning for O(N) Short-Range Force Computation.

Measured Innovation (Stream M6 / M5):
Reduces pair force evaluation complexity from O(N^2) to O(N) by partitioning the simulation
box into spatial grid cells of size >= cutoff.
"""

from __future__ import annotations

import math
import numpy as np


def pair_forces_cell_list(
    positions: np.ndarray,
    box_size: float,
    cutoff: float,
    spring_constant: float,
    equilibrium_distance: float,
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if box_size <= 0 or cutoff <= 0:
        raise ValueError("box_size and cutoff must be positive")

    n_particles = positions.shape[0]
    forces = np.zeros_like(positions)
    if n_particles < 2:
        return forces

    # Determine grid cell dimensions (cell size >= cutoff)
    num_cells_1d = max(1, int(math.floor(box_size / cutoff)))
    cell_size = box_size / num_cells_1d
    cutoff_sq = cutoff * cutoff

    # Assign particles to 3D grid cells
    cell_coords = np.floor(positions / cell_size).astype(int)
    np.clip(cell_coords, 0, num_cells_1d - 1, out=cell_coords)

    # Build cell buckets mapping cell (cx, cy, cz) tuple to list of particle indices
    cells: dict[tuple[int, int, int], list[int]] = {}
    for i in range(n_particles):
        key = (int(cell_coords[i, 0]), int(cell_coords[i, 1]), int(cell_coords[i, 2]))
        if key not in cells:
            cells[key] = []
        cells[key].append(i)

    # 26-neighbor cell offset vectors + self cell (13 unique relative directions to avoid double counting)
    neighbor_offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                # Unique direction filter to visit each cell pair once
                if (dx > 0) or (dx == 0 and dy > 0) or (dx == 0 and dy == 0 and dz >= 0):
                    neighbor_offsets.append((dx, dy, dz))

    # Evaluate forces between particles in adjacent cells
    for cell_key, p_indices in cells.items():
        cx, cy, cz = cell_key
        num_p = len(p_indices)

        for dx, dy, dz in neighbor_offsets:
            nx, ny, nz = cx + dx, cy + dy, cz + dz
            if not (0 <= nx < num_cells_1d and 0 <= ny < num_cells_1d and 0 <= nz < num_cells_1d):
                continue

            other_key = (nx, ny, nz)
            if other_key not in cells:
                continue

            q_indices = cells[other_key]
            same_cell = (dx == 0 and dy == 0 and dz == 0)

            for idx_i, i in enumerate(p_indices):
                start_j = (idx_i + 1) if same_cell else 0
                r_i = positions[i]

                for j in q_indices[start_j:]:
                    r_j = positions[j]
                    rx = r_i[0] - r_j[0]
                    ry = r_i[1] - r_j[1]
                    rz = r_i[2] - r_j[2]
                    dist_sq = rx * rx + ry * ry + rz * rz

                    if epsilon * epsilon < dist_sq <= cutoff_sq:
                        distance = math.sqrt(dist_sq)
                        scale = -spring_constant * (distance - equilibrium_distance) / distance
                        fx = scale * rx
                        fy = scale * ry
                        fz = scale * rz

                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[i, 2] += fz

                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
                        forces[j, 2] -= fz

    return forces
