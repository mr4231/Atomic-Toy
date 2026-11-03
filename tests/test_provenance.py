"""Tests for the reproducibility metadata attached to newly written results.

These tests protect the Phase 1 provenance contract: a saved artifact must be able
to say which code, environment and initial state produced it.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.model import SimulationConfig
from src.simulation.benchmarks import benchmark_protocol, save_benchmark_csv
from src.simulation.engine import create_random_state, run_simulation
from src.utils.io import load_result, save_result
from src.utils.provenance import (
    PROVENANCE_KEY, array_fingerprint, git_state, initial_condition_fingerprint,
    provenance_record,
)


class ProvenanceMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SimulationConfig(
            n_particles=3, box_length=20.0, cutoff=3.0, dt=0.01, steps=20, seed=7
        )
        self.state = create_random_state(self.config)
        self.result = run_simulation(self.config, self.state)

    def test_saved_json_contains_the_provenance_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stem = Path(temporary) / "run"
            _, json_path = save_result(self.result, stem, initial_state=self.state)
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertIn(PROVENANCE_KEY, metadata)
            block = metadata[PROVENANCE_KEY]
            for key in (
                "git_commit", "git_branch", "git_dirty", "python", "numpy", "platform",
                "initial_condition_sha256", "command_line", "written_utc",
            ):
                self.assertIn(key, block)
            self.assertEqual(
                block["initial_condition_sha256"], initial_condition_fingerprint(self.state)
            )

    def test_metadata_is_additive_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stem = Path(temporary) / "run"
            save_result(self.result, stem, initial_state=self.state)
            restored = load_result(stem)
            self.assertEqual(restored.config, self.config)
            np.testing.assert_allclose(restored.positions, self.result.positions)

    def test_missing_initial_state_is_recorded_as_null(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, json_path = save_result(self.result, Path(temporary) / "run")
            block = json.loads(json_path.read_text(encoding="utf-8"))[PROVENANCE_KEY]
            self.assertIsNone(block["initial_condition_sha256"])

    def test_fingerprint_is_deterministic_and_sensitive(self) -> None:
        first = initial_condition_fingerprint(self.state)
        second = initial_condition_fingerprint(create_random_state(self.config))
        self.assertEqual(first, second)
        perturbed = self.state.copy()
        perturbed.velocities[0, 0] += 1e-9
        self.assertNotEqual(first, initial_condition_fingerprint(perturbed))
        self.assertIsNone(array_fingerprint(None))

    def test_git_state_reports_the_revision(self) -> None:
        state = git_state()
        self.assertIsInstance(state["git_commit"], str)
        self.assertIsInstance(state["git_dirty"], bool)

    def test_provenance_record_extra_fields_are_merged(self) -> None:
        record = provenance_record(extra={"study": "unit-test"})
        self.assertEqual(record["study"], "unit-test")

    def test_benchmark_protocol_matches_the_required_comparison(self) -> None:
        protocol = benchmark_protocol()
        self.assertEqual(protocol["particle_counts"], [50, 200, 500, 1000])
        self.assertGreaterEqual(protocol["repeats"], 5)
        self.assertEqual(protocol["cutoff"], 2.5)
        self.assertEqual(protocol["warmup"], 0, "a missing warm-up must stay visible")

    def test_benchmark_csv_keeps_its_columns_and_writes_a_sidecar(self) -> None:
        rows = [{
            "implementation": "python_loop", "n_particles": 50, "repeats": 5,
            "median_seconds": 1e-4, "speedup_vs_python": 1.0, "force_parity_norm": 0.0,
        }]
        with tempfile.TemporaryDirectory() as temporary:
            path = save_benchmark_csv(rows, Path(temporary) / "performance" / "benchmark.csv")
            header = path.read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(header, ",".join(rows[0]))
            sidecar = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar["implementations"], ["python_loop"])
            self.assertEqual(sidecar["particle_counts"], [50])
            self.assertEqual(sidecar["protocol"]["cutoff"], 2.5)
            self.assertFalse(sidecar[PROVENANCE_KEY]["timing_reproducible"])

    def test_benchmark_sidecar_accepts_an_explicit_protocol(self) -> None:
        rows = [{
            "implementation": "numpy_vectorized", "n_particles": 50, "repeats": 5,
            "median_seconds": 1e-4, "speedup_vs_python": 1.0, "force_parity_norm": 0.0,
        }]
        with tempfile.TemporaryDirectory() as temporary:
            path = save_benchmark_csv(
                rows, Path(temporary) / "benchmark.csv", meta={"cutoff": 0.8, "warmup": 1}
            )
            sidecar = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar["protocol"], {"cutoff": 0.8, "warmup": 1})


    def test_config_fingerprint_is_recorded_and_parameter_sensitive(self) -> None:
        from dataclasses import replace

        from src.utils.provenance import config_fingerprint

        same = config_fingerprint(SimulationConfig(**self.config.as_dict()))
        self.assertEqual(config_fingerprint(self.config), same)
        self.assertNotEqual(
            config_fingerprint(replace(self.config, dt=0.02)), config_fingerprint(self.config)
        )
        self.assertIsNone(config_fingerprint(None))
        with tempfile.TemporaryDirectory() as temporary:
            _, json_path = save_result(
                self.result, Path(temporary) / "run", initial_state=self.state
            )
            block = json.loads(json_path.read_text(encoding="utf-8"))[PROVENANCE_KEY]
            self.assertEqual(block["config_sha256"], config_fingerprint(self.config))


if __name__ == "__main__":
    unittest.main()
