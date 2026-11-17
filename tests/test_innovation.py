
from __future__ import annotations

import unittest
import numpy as np

from src.physics import pair_forces_cell_list, pair_forces_loop, pair_forces_vectorized


class InnovationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.box_size = 10.0
        self.cutoff = 0.8
        self.spring_constant = 15.0
        self.equilibrium_distance = 0.8
        
        # Seeded random configuration of particles inside box
        rng = np.random.default_rng(42)
        self.positions = rng.uniform(0.0, self.box_size, (50, 3))

    def test_cell_list_matches_loop_and_vectorized_reference(self) -> None:
        ref_loop = pair_forces_loop(self.positions, self.cutoff, self.spring_constant, self.equilibrium_distance)
        ref_vec = pair_forces_vectorized(self.positions, self.cutoff, self.spring_constant, self.equilibrium_distance)
        cell_forces = pair_forces_cell_list(
            self.positions, self.box_size, self.cutoff, self.spring_constant, self.equilibrium_distance
        )

        np.testing.assert_allclose(cell_forces, ref_loop, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(cell_forces, ref_vec, rtol=1e-12, atol=1e-12)

    def test_cell_list_forces_sum_to_zero(self) -> None:
        cell_forces = pair_forces_cell_list(
            self.positions, self.box_size, self.cutoff, self.spring_constant, self.equilibrium_distance
        )
        np.testing.assert_allclose(cell_forces.sum(axis=0), np.zeros(3), atol=1e-12)

    def test_cell_list_close_pair_epsilon_convention(self) -> None:
        """Regression test for D-11 (fixed 2026-09-22).

        ``pair_forces_cell_list`` must use the same ``epsilon`` *distance*
        convention as the reference kernels and the C backend: pairs with
        ``distance <= epsilon`` are ignored, i.e. ``epsilon**2 < r**2`` must be
        tested against the squared distance.  The historical bug compared the
        unsquared ``epsilon`` against ``distance**2`` and silently dropped every
        pair with ``1e-12 < r <= 1e-6``; see
        results/audit_2026-09-22/cell_list/close_pair_epsilon_audit.csv and
        DECISION_LOG.md D-11.
        """
        # Separations spanning the old failure window (1e-12, 1e-5) and beyond.
        separations = [1e-13, 1e-12, 1e-11, 1e-9, 1e-7, 1e-6, 1e-5, 1e-3]
        for r in separations:
            with self.subTest(separation=r):
                positions = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
                ref = pair_forces_loop(
                    positions, self.cutoff, self.spring_constant, self.equilibrium_distance
                )
                cell = pair_forces_cell_list(
                    positions, self.box_size, self.cutoff,
                    self.spring_constant, self.equilibrium_distance,
                )
                if r <= 1e-12:
                    # Reference convention: distance <= epsilon -> pair ignored.
                    self.assertTrue(np.all(ref == 0.0))
                    self.assertTrue(np.all(cell == 0.0))
                else:
                    # Pair is active; cell list must now agree with the reference.
                    np.testing.assert_allclose(cell, ref, rtol=1e-12, atol=1e-12)
                    # Analytic harmonic force for r < r0: |F| = k*(r0 - r),
                    # valid at every active separation in this loop.
                    expected = self.spring_constant * (self.equilibrium_distance - r)
                    np.testing.assert_allclose(
                        np.linalg.norm(ref[0]), expected, rtol=1e-12
                    )
