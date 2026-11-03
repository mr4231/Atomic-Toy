"""Dashboard tests: read-only evidence access and protection of the frozen results.

The dashboard must never be able to overwrite ``results/``; these tests lock that
down and also check that the endpoints return saved data rather than invented values.
"""

from __future__ import annotations

import unittest
from pathlib import Path

try:
    import web_server
except ImportError as import_error:  # Flask is optional for the numerical tests
    web_server = None
    IMPORT_ERROR = import_error

REQUIRES_FLASK = unittest.skipUnless(web_server is not None, "flask is required")

if web_server is not None:
    ROOT_DIR = web_server.ROOT_DIR
    CANONICAL = web_server.CANONICAL_EVIDENCE_DIR
    RUN_ROOT = web_server.WEB_RUN_ROOT


def directory_snapshot(root: Path) -> set[str]:
    """Return the relative paths of every directory under ``root``."""
    if not root.exists():
        return set()
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir()}


@REQUIRES_FLASK
class DashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        web_server.app.config.update(TESTING=True)
        self.client = web_server.app.test_client()

    def test_home_page_renders(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_default_config_endpoint_matches_the_config_file(self) -> None:
        payload = self.client.get("/api/config/default").get_json()
        self.assertEqual(payload["number_of_particles"], 25)
        self.assertEqual(payload["cutoff"], 0.8)
        self.assertEqual(payload["initial_velocity_scale"], 8.0)

    def test_unknown_study_returns_404(self) -> None:
        self.assertEqual(self.client.get("/api/results/not-a-study").status_code, 404)

    def test_known_study_serves_the_frozen_rows(self) -> None:
        payload = self.client.get("/api/results/timestep").get_json()
        self.assertTrue(payload["exists"])
        self.assertEqual(len(payload["data"]), 4)

    def test_experiment_run_requires_confirmation_and_writes_nothing(self) -> None:
        before = (directory_snapshot(CANONICAL), directory_snapshot(RUN_ROOT))
        response = self.client.post("/api/experiments/run", json={"experiment": "memory"})
        self.assertEqual(response.status_code, 428)
        self.assertEqual(response.get_json()["status"], "confirmation_required")
        self.assertEqual((directory_snapshot(CANONICAL), directory_snapshot(RUN_ROOT)), before)

    def test_experiment_run_rejects_an_unknown_name(self) -> None:
        response = self.client.post(
            "/api/experiments/run", json={"experiment": "not-a-study", "confirm": True}
        )
        self.assertEqual(response.status_code, 400)

    def test_canonical_evidence_is_write_protected(self) -> None:
        with self.assertRaises(ValueError):
            web_server.assert_outside_canonical_evidence(CANONICAL / "memory" / "pairwise_memory.csv")
        with self.assertRaises(ValueError):
            web_server.assert_outside_canonical_evidence(CANONICAL)

    def test_new_run_directory_is_outside_the_evidence_tree(self) -> None:
        target = web_server.new_run_directory("memory")
        self.assertNotIn(CANONICAL.resolve(), target.resolve().parents)
        self.assertFalse(target.exists(), "the endpoint creates the run directory only on demand")

    def test_simulate_endpoint_returns_a_trajectory_without_writing_files(self) -> None:
        before = (directory_snapshot(CANONICAL), directory_snapshot(RUN_ROOT))
        response = self.client.post("/api/simulate", json={
            "number_of_particles": 3, "box_size": 20.0, "cutoff": 3.0,
            "spring_constant": 15.0, "equilibrium_distance": 0.8,
            "initial_velocity_scale": 2.0, "dt": 0.01, "steps": 5, "seed": 3,
        })
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(len(payload["positions"]), 6)
        self.assertEqual(len(payload["positions"][0]), 3)
        self.assertIn("max_relative_energy_drift", payload)
        self.assertEqual((directory_snapshot(CANONICAL), directory_snapshot(RUN_ROOT)), before)

    def test_simulate_endpoint_reports_invalid_configurations(self) -> None:
        response = self.client.post("/api/simulate", json={"number_of_particles": 0})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_simulate_endpoint_rejects_an_unbounded_response(self) -> None:
        response = self.client.post(
            "/api/simulate",
            json={"number_of_particles": 3, "steps": 5_001},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("at most", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
