
from __future__ import annotations

import unittest

import numpy as np

from src.model import SimulationConfig, SimulationState
from src.simulation.engine import run_simulation
from src.utils.metrics import verification_metrics


class DissipationTests(unittest.TestCase):
    def test_free_particle_velocity_matches_exact_linear_drag_decay(self) -> None:
        config = SimulationConfig(
            number_of_particles=1, box_size=100.0, cutoff=0.8,
            spring_constant=0.0, equilibrium_distance=0.8,
            dt=0.01, steps=100, gamma=0.7, integrator="velocity_verlet",
        )
        state = SimulationState(
            positions=np.array([[50.0, 50.0, 50.0]]),
            velocities=np.array([[3.0, -2.0, 1.0]]),
            masses=np.array([2.0]),
        )
        result = run_simulation(config, state)
        expected = state.velocities * np.exp(-config.gamma * config.steps * config.dt / 2.0)
        np.testing.assert_allclose(result.final_state.velocities, expected, rtol=1e-12, atol=1e-12)

    def test_verification_metrics_report_exact_basic_invariants(self) -> None:
        metrics = verification_metrics(
            np.array([[1.0, 1.0, 1.0], [1.5, 1.0, 1.0]]),
            box_size=10.0, cutoff=0.8, spring_constant=15.0, equilibrium_distance=0.8,
        )
        self.assertEqual(metrics["distance_symmetry_max_abs"], 0.0)
        self.assertEqual(metrics["distance_diagonal_max_abs"], 0.0)
        self.assertEqual(metrics["self_neighbor_count"], 0)
        self.assertEqual(metrics["max_box_violation"], 0.0)
        self.assertLess(metrics["net_internal_force_norm"], 1e-12)


if __name__ == "__main__":
    unittest.main()
