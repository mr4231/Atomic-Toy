"""Tests for the enforced output discipline around the frozen evidence tree.

These tests are read-only with respect to ``results/``: they assert that refusals
happen *before* any byte is written, and they verify the evidence tree is
byte-identical afterwards.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from src.model import SimulationConfig
from src.simulation.benchmarks import save_benchmark_csv
from src.simulation.engine import create_random_state, run_simulation
from src.simulation.experiments import memory_table
from src.utils.evidence_guard import (
    CANONICAL_EVIDENCE_ROOT, FROZEN_STUDY_DIRS, ProtectedEvidenceError, assert_safe_output,
    matched_frozen_dir,
)
from src.utils.io import save_result

ROOT = Path(__file__).resolve().parents[1]


def evidence_snapshot() -> dict[str, str]:
    """Return ``{relative path: sha256}`` for every file under results/."""
    snapshot: dict[str, str] = {}
    for path in sorted(CANONICAL_EVIDENCE_ROOT.rglob("*")):
        if path.is_file():
            snapshot[path.relative_to(CANONICAL_EVIDENCE_ROOT).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return snapshot


class EvidenceGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SimulationConfig(
            n_particles=2, box_length=20.0, cutoff=3.0, dt=0.01, steps=4, seed=1
        )
        self.state = create_random_state(self.config)
        self.result = run_simulation(self.config, self.state)

    def test_every_frozen_study_directory_is_detected(self) -> None:
        for directory in FROZEN_STUDY_DIRS:
            self.assertEqual(
                matched_frozen_dir(ROOT / directory / "some_new_file.csv"), directory
            )

    def test_a_new_output_root_is_allowed(self) -> None:
        candidate = ROOT / "results" / "timestep_s2" / "dt_0p0100.json"
        self.assertIsNone(matched_frozen_dir(candidate))
        self.assertEqual(assert_safe_output(candidate), candidate.resolve())

    def test_scratch_paths_outside_the_repository_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "baseline.npz"
            self.assertIsNone(matched_frozen_dir(target))
            self.assertEqual(assert_safe_output(target), target.resolve())

    def test_frozen_study_writes_are_refused(self) -> None:
        for directory in FROZEN_STUDY_DIRS:
            with self.assertRaises(ProtectedEvidenceError):
                assert_safe_output(ROOT / directory / "new-result.csv")

    def test_override_bypasses_the_refusal_explicitly(self) -> None:
        target = ROOT / "results" / "energy" / "baseline.json"
        self.assertEqual(
            assert_safe_output(target, allow_protected_overwrite=True), target.resolve()
        )

    def test_save_result_refuses_frozen_paths_without_writing(self) -> None:
        before = evidence_snapshot()
        with self.assertRaises(ProtectedEvidenceError):
            save_result(self.result, CANONICAL_EVIDENCE_ROOT / "energy" / "baseline")
        self.assertEqual(evidence_snapshot(), before)

    def test_save_result_still_writes_new_output_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            npz_path, json_path = save_result(self.result, Path(temporary) / "baseline")
            self.assertTrue(npz_path.exists() and json_path.exists())

    def test_benchmark_csv_refuses_the_frozen_performance_directory(self) -> None:
        rows = [{
            "implementation": "python_loop", "n_particles": 50, "repeats": 5,
            "median_seconds": 1e-4, "speedup_vs_python": 1.0, "force_parity_norm": 0.0,
        }]
        before = evidence_snapshot()
        with self.assertRaises(ProtectedEvidenceError):
            save_benchmark_csv(rows, CANONICAL_EVIDENCE_ROOT / "performance" / "benchmark.csv")
        self.assertEqual(evidence_snapshot(), before)

    def test_summary_csv_writers_refuse_frozen_directories(self) -> None:
        before = evidence_snapshot()
        with self.assertRaises(ProtectedEvidenceError):
            memory_table(CANONICAL_EVIDENCE_ROOT / "memory")
        self.assertEqual(evidence_snapshot(), before)

    def test_existing_scratch_file_is_left_untouched(self) -> None:
        # Rule 2 (an existing file under results/) is currently unreachable through a
        # frozen directory, because every file that existed at freeze time lives in one:
        # results/ holds only the six frozen study directories. The check below therefore
        # covers the scratch case, which must never be blocked.
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary) / "already-there.csv"
            scratch.write_text("x", encoding="utf-8")
            assert_safe_output(scratch)
            self.assertEqual(scratch.read_text(encoding="utf-8"), "x")

    def test_guard_module_is_exported_from_the_utils_package(self) -> None:
        import src.utils as utils

        self.assertIn("assert_safe_output", utils.__all__)
        self.assertIn("config_fingerprint", utils.__all__)


if __name__ == "__main__":
    unittest.main()
