
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.model import SimulationConfig, SimulationState
from src.simulation.engine import create_random_state, run_simulation
from src.simulation.experiments import gamma_sweep, precision_comparison, timestep_sweep
from src.utils.io import load_result, save_result
from src.utils.metrics import pairwise_storage_bytes


class SimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SimulationConfig(n_particles=2, box_length=20.0, cutoff=3.0, spring_constant=10.0,
                                       equilibrium_distance=0.8, dt=0.01, steps=100, seed=4)
        self.state = SimulationState(
            np.array([[9.5, 10.0, 10.0], [10.5, 10.0, 10.0]]),
            np.array([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]]), np.ones(2),
        )

    def test_simulation_stays_in_box_and_records_all_diagnostics(self) -> None:
        result = run_simulation(self.config, self.state)
        self.assertEqual(result.positions.shape, (101, 2, 3))
        self.assertEqual(len(result.times), 101)
        self.assertTrue(np.all(result.positions >= 0.0))
        self.assertTrue(np.all(result.positions <= self.config.box_length))
        self.assertTrue(np.all(np.isfinite(result.total_energy)))

    def test_velocity_verlet_has_no_worse_energy_drift_than_baseline_euler(self) -> None:
        verlet = run_simulation(replace(self.config, integrator="velocity_verlet"), self.state)
        euler = run_simulation(replace(self.config, integrator="semi_implicit_euler"), self.state)
        self.assertLessEqual(verlet.max_relative_energy_drift, euler.max_relative_energy_drift + 1e-9)

    def test_gamma_zero_reproduces_conservative_case(self) -> None:
        first = run_simulation(replace(self.config, gamma=0.0), self.state)
        second = run_simulation(replace(self.config, gamma=0.0), self.state)
        np.testing.assert_allclose(first.positions, second.positions)
        np.testing.assert_allclose(first.total_energy, second.total_energy)

    def test_result_round_trip_preserves_arrays_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stem = Path(temporary) / "run"
            original = run_simulation(self.config, self.state)
            save_result(original, stem)
            restored = load_result(stem)
            np.testing.assert_allclose(restored.positions, original.positions)
            self.assertEqual(restored.config, original.config)

    def test_memory_formula_matches_dense_array_size(self) -> None:
        dense = np.zeros((10, 10, 3), dtype=np.float32)
        self.assertEqual(pairwise_storage_bytes(10, "float32"), dense.nbytes)

    def test_random_initialization_is_reproducible_for_a_seed(self) -> None:
        config = SimulationConfig(number_of_particles=24, box_size=20.0, seed=31)
        first = create_random_state(config)
        second = create_random_state(config)
        np.testing.assert_array_equal(first.positions, second.positions)
        np.testing.assert_array_equal(first.velocities, second.velocities)

    def test_random_initial_positions_are_inside_the_box_and_separated(self) -> None:
        config = SimulationConfig(
            number_of_particles=30, box_size=20.0, minimum_initial_separation=0.7, seed=5
        )
        state = create_random_state(config)
        self.assertTrue(np.all(state.positions >= 0.0))
        self.assertTrue(np.all(state.positions <= config.box_size))
        distances = np.linalg.norm(state.positions[:, None, :] - state.positions[None, :, :], axis=2)
        unique_distances = distances[np.triu_indices(config.number_of_particles, k=1)]
        self.assertGreaterEqual(float(np.min(unique_distances)), config.minimum_initial_separation)

    def test_initial_velocities_are_nonzero_and_follow_the_configured_scale(self) -> None:
        config = SimulationConfig(
            number_of_particles=200, box_size=30.0, initial_velocity_scale=2.0, seed=9
        )
        state = create_random_state(config)
        self.assertTrue(np.all(np.linalg.norm(state.velocities, axis=1) > 0.0))
        # Component RMS converges to the configured normal-distribution standard deviation.
        component_rms = float(np.sqrt(np.mean(state.velocities * state.velocities)))
        self.assertAlmostEqual(component_rms, config.initial_velocity_scale, delta=0.25)

    def test_small_experiment_runners_create_summary_csvs(self) -> None:
        small = replace(self.config, steps=20, frame_stride=2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertTrue(timestep_sweep(small, root / "timestep", (0.02, 0.01)).exists())
            self.assertTrue(gamma_sweep(small, root / "gamma", (0.0, 0.1)).exists())
            self.assertTrue(precision_comparison(small, root / "precision").exists())
