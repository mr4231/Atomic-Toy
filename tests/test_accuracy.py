"""Focused tests for the Gate-G3 accuracy-study machinery (src/simulation/accuracy.py).

These tests run tiny studies into temporary directories only; the official G3
experiment (data/configs/accuracy_interacting.json into a dated results root)
has NOT been executed here.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.model import SimulationConfig, SimulationState
from src.simulation.accuracy import (
    aligned_samples,
    frame_indices_for_times,
    measured_orders,
    run_accuracy_study,
    sample_times,
)
from src.simulation.engine import create_random_state
from src.utils.provenance import config_fingerprint, initial_condition_fingerprint


def _tiny_config(**overrides) -> SimulationConfig:
    """A small gamma-free configuration with an exact horizon of 1.0."""
    values = dict(
        number_of_particles=4,
        box_size=10.0,
        cutoff=0.8,
        spring_constant=15.0,
        equilibrium_distance=0.8,
        initial_velocity_scale=2.0,
        minimum_initial_separation=0.5,
        dt=0.05,
        steps=20,
        gamma=0.0,
        integrator="velocity_verlet",
        dtype="float64",
        seed=0,
        frame_stride=1,
    )
    values.update(overrides)
    return SimulationConfig(**values)


class AccuracyStudyTests(unittest.TestCase):
    def test_identical_initial_condition_across_all_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = run_accuracy_study(
                _tiny_config(), tmp, dts=(0.1, 0.05),
                reference_dt=0.0125, refinement_dt=0.00625,
            )
            recorded = manifest["initial_condition_sha256"]
            self.assertTrue(recorded)
            self.assertEqual(
                recorded, initial_condition_fingerprint(create_random_state(_tiny_config()))
            )
            first_positions = None
            entries = manifest["cases"] + [manifest["reference"], manifest["reference_refinement"]]
            for entry in entries:
                metadata = json.loads(
                    Path(entry["output_stem"] + ".json").read_text(encoding="utf-8")
                )
                self.assertEqual(metadata["provenance"]["initial_condition_sha256"], recorded)
                with np.load(entry["output_stem"] + ".npz") as arrays:
                    if first_positions is None:
                        first_positions = arrays["positions"][0]
                    else:
                        np.testing.assert_array_equal(arrays["positions"][0], first_positions)

    def test_common_sample_times_are_present_in_every_recorded_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = run_accuracy_study(
                _tiny_config(), tmp, dts=(0.1, 0.05),
                reference_dt=0.0125, refinement_dt=0.00625,
            )
            wanted = sample_times(manifest["horizon"], manifest["sample_interval"])
            self.assertEqual(len(wanted), 11)
            entries = manifest["cases"] + [manifest["reference"], manifest["reference_refinement"]]
            for entry in entries:
                with np.load(entry["output_stem"] + ".npz") as arrays:
                    indices = frame_indices_for_times(arrays["frame_times"], wanted)
                    np.testing.assert_allclose(arrays["frame_times"][indices], wanted, atol=1e-9)

    def test_frame_indices_raise_for_a_missing_sample_time(self) -> None:
        with self.assertRaises(ValueError):
            frame_indices_for_times(np.arange(3) * 0.1, [0.05])

    def test_default_sample_interval_is_the_common_multiple_grid(self) -> None:
        # 0.05 is NOT a multiple of 0.02, so the common grid must be the least
        # common multiple (0.1 here), not the coarsest dt.
        with tempfile.TemporaryDirectory() as tmp:
            manifest = run_accuracy_study(
                _tiny_config(), tmp, dts=(0.05, 0.02),
                reference_dt=0.0125, refinement_dt=0.00625,
            )
            self.assertAlmostEqual(manifest["sample_interval"], 0.1, places=12)
            self.assertEqual(manifest["n_sample_times"], 11)

    def test_euler_and_verlet_position_trajectories_differ_for_an_interacting_case(self) -> None:
        # Regression for the G3 review anomaly: on a genuinely interacting
        # configuration (a close harmonic pair), the two integrators MUST produce
        # different position trajectories.  If the engine ever reused/shared the
        # case trajectory (or the metric compared case-vs-case), this test would
        # fail.  It also pins that the difference is physical (grows with time)
        # rather than round-off noise.
        config = _tiny_config(number_of_particles=2, minimum_initial_separation=0.1)
        state = SimulationState(
            np.array([[4.7, 5.0, 5.0], [5.3, 5.0, 5.0]]),
            np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
            np.ones(2),
        )
        with tempfile.TemporaryDirectory() as tmp:
            manifest = run_accuracy_study(
                config, tmp, dts=(0.05, 0.025), sample_interval=0.05,
                reference_dt=0.00625, refinement_dt=0.003125, initial_state=state,
            )
            stems = {
                (row["integrator"], row["dt"]): row["output_stem"]
                for row in manifest["cases"]
            }
            with np.load(stems[("semi_implicit_euler", 0.05)] + ".npz") as e, \
                    np.load(stems[("velocity_verlet", 0.05)] + ".npz") as v:
                diff = np.abs(e["positions"] - v["positions"])
                self.assertGreater(float(diff.max()), 1e-6)
                # Positions agree at t=0 (same IC) and diverge afterwards.
                self.assertEqual(float(diff[0].max()), 0.0)
                self.assertGreater(float(diff[-1].max()), 1e-6)
                # Velocities must differ too (the integrators are not aliased).
                self.assertGreater(
                    float(np.abs(e["final_velocities"] - v["final_velocities"]).max()), 1e-6
                )

    def test_position_error_compares_each_case_to_the_reference(self) -> None:
        # The stored position_error must equal an independent recomputation of
        # ||case(t) - reference(t)||/scale over the common times, per integrator,
        # and the two integrators' errors must differ on an interacting case
        # (they would coincide if the metric reused one trajectory for both).
        config = _tiny_config(number_of_particles=2, minimum_initial_separation=0.1)
        state = SimulationState(
            np.array([[4.7, 5.0, 5.0], [5.3, 5.0, 5.0]]),
            np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
            np.ones(2),
        )
        with tempfile.TemporaryDirectory() as tmp:
            manifest = run_accuracy_study(
                config, tmp, dts=(0.05, 0.025), sample_interval=0.05,
                reference_dt=0.00625, refinement_dt=0.003125, initial_state=state,
            )
            wanted = sample_times(manifest["horizon"], manifest["sample_interval"])
            with np.load(manifest["reference"]["output_stem"] + ".npz") as ref_npz:
                ref = SimpleNamespace(positions=ref_npz["positions"], frame_times=ref_npz["frame_times"])
            ref_positions, ref_times = aligned_samples(ref, wanted)
            errors = {}
            for row in manifest["cases"]:
                with np.load(row["output_stem"] + ".npz") as case_npz:
                    case = SimpleNamespace(positions=case_npz["positions"], frame_times=case_npz["frame_times"])
                case_positions, case_times = aligned_samples(case, wanted)
                self.assertTrue(np.allclose(case_times, ref_times, atol=1e-9))
                recomputed = max(
                    float(np.linalg.norm((c - r).ravel()))
                    for c, r in zip(case_positions, ref_positions)
                ) / (config.box_size * np.sqrt(config.n_particles))
                self.assertAlmostEqual(recomputed, row["position_error"], places=15)
                errors[(row["integrator"], row["dt"])] = row["position_error"]
            # Interacting case: the integrators must not report identical errors.
            self.assertNotAlmostEqual(
                errors[("semi_implicit_euler", 0.05)],
                errors[("velocity_verlet", 0.05)],
                places=6,
            )
            self.assertNotAlmostEqual(
                errors[("semi_implicit_euler", 0.025)],
                errors[("velocity_verlet", 0.025)],
                places=6,
            )

    def test_reference_comparison_and_refinement_on_a_wall_free_two_body_system(self) -> None:
        # Two particles oscillating through the harmonic well, centred so that no
        # wall is ever reached: the reference comparison is then purely about the
        # timestep, not about the wall rule.
        config = _tiny_config(number_of_particles=2, minimum_initial_separation=0.1)
        state = SimulationState(
            np.array([[4.7, 5.0, 5.0], [5.3, 5.0, 5.0]]),
            np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
            np.ones(2),
        )
        with tempfile.TemporaryDirectory() as tmp:
            manifest = run_accuracy_study(
                config, tmp, dts=(0.05, 0.025), sample_interval=0.05,
                reference_dt=0.00625, refinement_dt=0.003125, initial_state=state,
            )
            by_integrator: dict[str, dict[float, float]] = {}
            for row in manifest["cases"]:
                by_integrator.setdefault(row["integrator"], {})[row["dt"]] = row["position_error"]
            for integrator, errors in by_integrator.items():
                self.assertGreater(errors[0.05], errors[0.025], integrator)
                self.assertGreater(errors[0.025], 0.0, integrator)
            consistency = manifest["reference_self_consistency"]["position_error"]
            self.assertGreaterEqual(consistency, 0.0)
            self.assertLess(consistency, by_integrator["velocity_verlet"][0.025])

    def test_measured_orders_from_synthetic_power_laws(self) -> None:
        second_order = measured_orders((0.2, 0.1, 0.05), (4e-3, 1e-3, 2.5e-4), "velocity_verlet")
        self.assertEqual(len(second_order), 3)  # two adjacent pairs + one overall
        for row in second_order:
            self.assertAlmostEqual(row["order"], 2.0, places=9)
        first_order = measured_orders((0.2, 0.1), (2e-3, 1e-3), "semi_implicit_euler")
        for row in first_order:
            self.assertAlmostEqual(row["order"], 1.0, places=9)
        zero_fine = measured_orders((0.2, 0.1), (1e-3, 0.0), "velocity_verlet")
        self.assertIsNone(zero_fine[0]["order"])
        inverted = measured_orders((0.2, 0.1), (1e-3, 2e-3), "semi_implicit_euler")
        self.assertLess(inverted[0]["order"], 0.0)

    def test_study_writes_expected_artifacts_and_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_accuracy_study(
                _tiny_config(), tmp, dts=(0.1, 0.05),
                reference_dt=0.0125, refinement_dt=0.00625,
            )
            root = Path(tmp)
            expected_files = (
                "accuracy_summary.csv", "convergence_orders.csv", "accuracy_manifest.json",
                "reference.npz", "reference.json",
                "reference_refinement.npz", "reference_refinement.json",
                "dt_0p1000_euler.npz", "dt_0p1000_verlet.npz",
                "dt_0p0500_euler.npz", "dt_0p0500_verlet.npz",
            )
            for name in expected_files:
                self.assertTrue((root / name).exists(), name)
            with (root / "accuracy_summary.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            required = {
                "dt", "steps", "integrator", "gamma", "dtype", "seed", "ic_sha256",
                "reference_config_sha256", "t_end", "max_relative_energy_drift",
                "position_error", "final_velocity_error",
            }
            self.assertTrue(required.issubset(rows[0].keys()))

    def test_manifest_records_reference_fingerprints_and_enforces_gamma_zero(self) -> None:
        base = _tiny_config()
        with tempfile.TemporaryDirectory() as tmp:
            manifest = run_accuracy_study(
                base, tmp, dts=(0.1, 0.05), reference_dt=0.0125, refinement_dt=0.00625
            )
            expected_reference = replace(
                base, dt=0.0125, steps=80, integrator="velocity_verlet", frame_stride=8
            )
            self.assertEqual(
                manifest["reference_config_sha256"], config_fingerprint(expected_reference)
            )
            self.assertEqual(manifest["n_sample_times"], 11)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_accuracy_study(_tiny_config(gamma=0.1), tmp)


if __name__ == "__main__":
    unittest.main()
