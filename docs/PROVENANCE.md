# Provenance of the stored experiment results

**Checker:** `python -B -m tools.check_provenance --date 2026-09-22`
**Machine-readable table:** `evidence_frozen/2026-09-22/PROVENANCE.csv`
**Immutable frozen copy of the evidence:** `evidence_frozen/2026-09-22/` (read-only)

Environment of the check: Windows 11, Python 3.14.0, numpy 2.5.3, matplotlib 3.11.2,
pillow 12.3.0, Flask 3.1.3, gcc 16.1.0 (`evidence_frozen/2026-09-22/ENVIRONMENT.txt`).

## What the checker decides

| Verdict | Meaning |
|---|---|
| `REPRODUCIBLE` | The recorded configuration regenerates the stored initial positions **and** re-integrating it in memory reproduces the stored arrays. |
| `LEGACY_IC_MISMATCH` | The recorded configuration does **not** regenerate the stored initial positions. No stored initial state exists, so the run cannot be replayed. |
| `FORMULA_MATCH` | Deterministic arithmetic artifact recomputed exactly. |
| `PARITY_ONLY` | Timing columns are machine dependent; the recorded force-parity norms and benchmark inputs were recomputed instead. |

Tolerance for the trajectory/IC comparisons: 1e-9 (absolute). The checker never writes
inside `results/`.

**Wall-rule caveat (D-06, decided 2026-09-22):** the `baseline` replay in the table above
was performed with the **clamped** wall rule that produced the stored file. After the D-06
decision, `reflect_walls` implements exact specular mirror/fold, and a replay of the same
recorded configuration now yields the mirror trajectory (drift **1.7203835e-04**, matching
`results/audit_2026-09-22/wall/wall_drift_comparison.csv`). The initial conditions are
unaffected (`create_random_state` does not involve walls), so the **IC part of the verdict
stands bit-identically**; the trajectory comparison now differs **by documented design**,
not by a code regression. Any future checker run must be read together with
`DECISION_LOG.md` D-06.

## Verdict table (measured 2026-09-22)

| Artifact | Study | N | cutoff | dt | frames | IC max abs diff | trajectory max abs diff | stored E0 | rerun E0 | stored drift | rerun drift | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` | energy | 25 | 0.8 | 0.01 | 501 | **0.0** | **0.0** | 2300.94486 | 2300.94486 | 2.3909296e-04 | 2.3909296e-04 | **REPRODUCIBLE** |
| `dt_0p0100` | timestep | 12 | 2.5 | 0.01 | 501 | 8.1119693 | 8.7953009 | 0.2748937 | 120.44378 | 8.0576873e-05 | 2.1979638 | LEGACY_IC_MISMATCH |
| `dt_0p0050` | timestep | 12 | 2.5 | 0.005 | 1001 | 8.1119693 | 8.5011593 | 0.2748937 | 120.44378 | 2.0129772e-05 | 2.3582645 | LEGACY_IC_MISMATCH |
| `dt_0p0200` | timestep | 12 | 2.5 | 0.02 | 251 | 8.1119693 | 8.4136500 | 0.2748937 | 120.44378 | 3.2267569e-04 | 2.1597276 | LEGACY_IC_MISMATCH |
| `dt_0p0500` | timestep | 12 | 2.5 | 0.05 | 101 | 8.1119693 | 8.4378674 | 0.2748937 | 120.44378 | 2.0633689e-03 | 1.4347746 | LEGACY_IC_MISMATCH |
| `gamma_0p000/0p050/0p100/0p500` | dissipation | 12 | 2.5 | 0.01 | 501 | 8.1119693 | 8.7953389 | 0.2748936 | 120.44379 | stored only | — | LEGACY_IC_MISMATCH |
| `float32` | precision | 12 | 2.5 | 0.01 | 501 | 8.1119692 | 8.7953389 | 0.2748936 | 120.44379 | 8.4888183e-05 | 2.1979871 | LEGACY_IC_MISMATCH |
| `float64` | precision | 12 | 2.5 | 0.01 | 501 | 8.1119693 | 8.7953009 | 0.2748937 | 120.44378 | 8.0576873e-05 | 2.1979638 | LEGACY_IC_MISMATCH |
| `pairwise_memory.csv` | memory | 100-10000 | — | — | — | — | 0 (exact) | — | — | — | — | **FORMULA_MATCH** (8/8 rows) |
| `benchmark.csv` | performance | 50-1000 | 2.5 | — | — | — | parity diff 0.0 | — | — | — | — | **PARITY_ONLY** |
| `benchmark.csv` (`c_optimized` rows) | performance | 50-1000 | 2.5 | — | — | — | — | — | — | — | — | NOT_CHECKED (C library deliberately not rebuilt) |

### Why the legacy rows are unreproducible

* The recorded configuration files predate the canonical field names: they contain
  `n_particles`/`box_length` and **omit `number_of_particles`, `box_size`,
  `initial_velocity_scale` and `minimum_initial_separation`** (the four fields the
  checker reports as missing). The current generator therefore cannot be reconstructed
  from the metadata.
* The stored runs have `E0 = 0.2748937`, whereas the same recorded parameters
  (`N = 12`, `cutoff = 2.5`, `dt = 0.01`, 500 steps, seed 0) produce `E0 = 120.44378`
  with the current code — evidence of a different initial placement rule and a much
  smaller initial velocity scale.
* All ten legacy artifacts share the identical `IC max abs diff = 8.1119693`, i.e. they
  were produced by one earlier generator that no longer exists.
* No stored initial state is written to disk, so those runs cannot be replayed.

### Classification of every existing artifact

| Category | Artifacts | Action |
|---|---|---|
| **Reproducible current evidence** | `results/energy/baseline.{npz,json}` (initial conditions and trajectory reproduce **bit-identically**); `results/memory/pairwise_memory.csv` (formula recomputed exactly, 8/8 rows); the `force_parity_norm` column of `results/performance/benchmark.csv` (recomputed equal to recorded: 2.248e-14, 1.632e-13, 6.506e-13, 2.247e-12 for the cell list; 0.0 for the loop and NumPy backends) | May be quoted. The benchmark's wall-clock column may **not** be quoted as reproducible: timings are machine dependent and must be re-measured with provenance. |
| **Legacy / unreproducible evidence** | `results/timestep/*`, `results/dissipation/*`, `results/precision/*` (all dated 17-09) and their summary CSVs; `report/FINAL_REPORT.md` §§3–4 tables; the dashboard panels for those three studies | Must not be quoted as current. Retained unchanged in the frozen snapshot for audit; to be superseded by the Phase 3–4 studies. No stored initial state exists, so they cannot be replayed. |
| **Stale derived artifacts** | `report/figures/baseline_energy.png` and `visualisation/baseline.gif` (both predate the 25-particle baseline); `report/figures/performance.png` (**missing**, though the README documents its generation command); the measured-value strings hard-coded in `templates/index.html` | Derived, not measurements. Do not delete (they are frozen in the snapshot), but regenerate from new-root runs before the report is final, and replace the hard-coded strings with values rendered from the saved files. |

## Policy for everything written from now on

1. New runs write to a **new output root** (`--output-root`, `--output`), never into
   `results/`.
2. Every new JSON/NPZ/CSV sidecar carries a `provenance` block: git commit, dirty flag,
   branch, Python/NumPy/platform, the **initial-condition SHA-256**, the **configuration
   SHA-256**, the command line and the UTC write time (`src/utils/provenance.py`, wired into
   `src/utils/io.py`). The configuration digest is computed over the canonical JSON of the
   recorded parameters, so two runs can be compared without diffing whole files.
3. Timing artifacts additionally carry a sidecar protocol record
   (`src/simulation/benchmarks.py`) — timings are never claimed as reproducible.
4. The rule is **enforced, not advisory**: `src/utils/evidence_guard.py` raises
   `ProtectedEvidenceError` before any byte is written when a target is inside a frozen
   study directory (`results/energy`, `results/timestep`, `results/dissipation`,
   `results/precision`, `results/memory`, `results/performance`) or when it would
   overwrite an existing file under `results/`. Scratch and temporary paths are unaffected.
5. Consequence to expect: the previously documented commands that defaulted into
   `results/<study>` (for example `python -m src.simulation.experiments timestep` or a bare
   `python run_demo.py`) now stop with an explanatory error instead of overwriting frozen
   evidence. Pass `--output-root results/<study>_<date>` or `--output <new stem>`.
   The bypass (`allow_protected_overwrite=True`) exists only in code, never on a command
   line, so a replacement must be deliberate.
6. Re-running `tools/check_provenance.py` against an existing snapshot writes
   `PROVENANCE.csv` inside that snapshot directory; use a new `--date` for a new check.
