"""Required numerical checks for geometry, forces, energy, and hard walls."""

from __future__ import annotations

import unittest

import numpy as np

from src.physics import (
    kinetic_energy, pair_forces_loop, pair_forces_vectorized, pairwise_geometry,
    potential_energy, reflect_walls,
)


class PhysicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.positions = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [4.0, 0.0, 0.0],
        ])

    def test_distances_are_symmetric_and_diagonal_is_zero(self) -> None:
        geometry = pairwise_geometry(self.positions, cutoff=2.1)
        np.testing.assert_allclose(geometry.distance, geometry.distance.T)
        np.testing.assert_array_equal(np.diag(geometry.distance), np.zeros(4))

    def test_neighbor_mask_has_no_self_neighbor(self) -> None:
        geometry = pairwise_geometry(self.positions, cutoff=2.1)
        self.assertFalse(np.any(np.diag(geometry.neighbors)))
        self.assertTrue(geometry.neighbors[0, 1])
        self.assertFalse(geometry.neighbors[0, 3])

    def test_contact_cutoff_excludes_pairs_outside_equilibrium_distance(self) -> None:
        positions = np.array([[0.0, 0.0, 0.0], [0.79, 0.0, 0.0], [0.81, 0.0, 0.0]])
        geometry = pairwise_geometry(positions, cutoff=0.8)
        self.assertTrue(geometry.neighbors[0, 1])
        self.assertFalse(geometry.neighbors[0, 2])

    def test_loop_and_vectorized_forces_match_and_sum_to_zero(self) -> None:
        loop = pair_forces_loop(self.positions, 2.1, 10.0, 0.8)
        vectorized = pair_forces_vectorized(self.positions, 2.1, 10.0, 0.8)
        np.testing.assert_allclose(loop, vectorized, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(loop.sum(axis=0), np.zeros(3), atol=1e-12)

    def test_hand_checked_pair_force_and_potential(self) -> None:
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        forces = pair_forces_loop(positions, 2.0, 10.0, 0.8)
        np.testing.assert_allclose(forces, [[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
        self.assertAlmostEqual(potential_energy(positions, 2.0, 10.0, 0.8), 0.2)

    def test_wall_reflection_mirrors_only_the_crossed_component(self) -> None:
        # One upper-wall crossing in x and one lower-wall crossing in z.
        positions = np.array([[10.1, 5.0, -0.2]])
        velocities = np.array([[2.0, 3.0, -4.0]])
        reflected_positions, reflected_velocities = reflect_walls(positions, velocities, 10.0)
        # x: 10.1 -> mirror to 9.9, vx flips; z: -0.2 -> mirror to 0.2, vz flips;
        # y is untouched (5.0, vy = +3).
        np.testing.assert_allclose(reflected_positions, [[9.9, 5.0, 0.2]])
        np.testing.assert_allclose(reflected_velocities, [[-2.0, 3.0, 4.0]])

    def test_wall_reflection_is_exact_specular_for_a_single_crossing(self) -> None:
        # Analytic case from the G2.1 audit: x0 = 9.9, v = 5, dt = 0.1 -> 10.0 overshoot,
        # exact specular maps the endpoint 10.4 to 2L - 10.4 = 9.6 (clamp gave 10.0).
        positions = np.array([[10.4, 0.0, 0.0]])
        velocities = np.array([[5.0, 0.0, 0.0]])
        reflected_positions, reflected_velocities = reflect_walls(positions, velocities, 10.0)
        np.testing.assert_allclose(reflected_positions, [[9.6, 0.0, 0.0]])
        np.testing.assert_allclose(reflected_velocities, [[-5.0, 0.0, 0.0]])

    def test_wall_reflection_folds_multiple_crossings(self) -> None:
        # x0 = 5.0, v = 200, displacement 20 = exactly two crossings: endpoint 25.0
        # folds to 5.0 with the velocity still +200 (even crossing count).
        positions = np.array([[25.0, 3.0, 0.0]])
        velocities = np.array([[200.0, 1.0, 0.0]])
        reflected_positions, reflected_velocities = reflect_walls(positions, velocities, 10.0)
        np.testing.assert_allclose(reflected_positions, [[5.0, 3.0, 0.0]])
        np.testing.assert_allclose(reflected_velocities, [[200.0, 1.0, 0.0]])

        # Three crossings (x0 = 3.0, v = 300, endpoint 33.0): odd count -> mirror to
        # 7.0 with the velocity reversed.
        positions = np.array([[33.0, 0.0, 0.0]])
        velocities = np.array([[300.0, 0.0, 0.0]])
        reflected_positions, reflected_velocities = reflect_walls(positions, velocities, 10.0)
        np.testing.assert_allclose(reflected_positions, [[7.0, 0.0, 0.0]])
        np.testing.assert_allclose(reflected_velocities, [[-300.0, 0.0, 0.0]])

    def test_wall_reflection_is_time_reversible(self) -> None:
        # Forward n steps, negate all velocities, forward n steps again, then
        # undo the deliberate reversal: the exact specular rule must return the
        # initial state, including momenta (the old clamp left a 0.2 position
        # error here, and retracing with a single negation leaves velocities
        # flipped because each retraced crossing flips them once more).
        box = 10.0
        positions = np.array([[9.9, 0.2, 5.0], [0.1, 9.8, 0.3]])
        velocities = np.array([[5.0, 1.0, -2.0], [-4.0, -1.0, 3.0]])
        start_positions = positions.copy()
        start_velocities = velocities.copy()
        dt = 0.05
        for _ in range(20):
            positions += velocities * dt
            reflect_walls(positions, velocities, box)
        velocities *= -1.0
        for _ in range(20):
            positions += velocities * dt
            reflect_walls(positions, velocities, box)
        velocities *= -1.0  # undo the reversal; reflection flips must cancel too
        np.testing.assert_allclose(positions, start_positions, atol=1e-12)
        np.testing.assert_allclose(velocities, start_velocities, atol=1e-12)

    def test_kinetic_energy(self) -> None:
        self.assertAlmostEqual(kinetic_energy(np.array([[3.0, 4.0, 0.0]]), np.array([2.0])), 25.0)
