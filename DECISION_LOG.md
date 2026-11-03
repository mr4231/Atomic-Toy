# Decision Log

Every scientific or structural decision that affects the interpretation of the
project's evidence is recorded here. Until an entry is marked **DECIDED** with an
evidence reference, the corresponding behaviour must not be changed.

Status values: `PENDING` (no decision), `DECIDED (process)` (a workflow rule, no physics
change), `DECIDED` (a choice backed by the evidence named in the row).

Rule: a physics change requires a measurement, an entry here, and an update to
`PROJECT_ARCHITECTURE.md` in the same change.

---

## 1. Workflow and reproducibility (D-01 – D-05)

| ID | Date | Status | Decision / open question | Owner | Evidence required |
|---|---|---|---|---|---|
| D-01 | 2026-09-22 | **DECIDED (process)** | The live evidence tree `results/` is frozen read-only. Every existing artifact is preserved in `evidence_frozen/2026-09-22/` with a SHA-256 manifest, environment record and git state. | M1 | `evidence_frozen/2026-09-22/{MANIFEST.csv,ENVIRONMENT.txt,GIT_STATE.txt,working_tree.patch}` |
| D-02 | 2026-09-22 | **DECIDED (process)** | `data/default_config.json` is not edited. New studies use a versioned copy under `data/configs/`. | M1 | `docs/PROVENANCE.md` |
| D-03 | 2026-09-22 | **DECIDED (process)** | New runs write to a new output root only. The dashboard must never write inside `results/`, and the rule is **enforced** by `src/utils/evidence_guard.py`: writers raise `ProtectedEvidenceError` for the frozen study directories (`results/energy`, `results/timestep`, `results/dissipation`, `results/precision`, `results/memory`, `results/performance`) and for any existing file under `results/`. Temporary paths and `runs/` are unaffected. Documented in `docs/PROVENANCE.md`; the freeze-branch copy of this row predates the guard. | M1 / M6 | `web_server.py` guard, `src/utils/evidence_guard.py`, `tests/test_dashboard.py`, `tests/test_evidence_guard.py` |
| D-04 | 2026-09-22 | **DECIDED (process)** | Every newly written result carries a provenance block (git commit, dirty flag, branch, versions, platform, **initial-condition SHA-256**, **configuration SHA-256**, command line, UTC time). | M1 | `src/utils/provenance.py`, `src/utils/io.py`, `src/simulation/benchmarks.py` sidecar |
| D-05 | 2026-09-22 | **DECIDED (process)** | Wall-clock timings are treated as machine dependent and are never quoted as reproducible evidence; only force-parity norms and input fingerprints are. | M5 | `docs/PROVENANCE.md` |

## 2. Physics questions (D-06 – D-11)

D-06, D-08 and D-11 were decided on 2026-09-22 from the Gate-G2 measurements in
`results/audit_2026-09-22/`. D-07 and D-10 remain PENDING.

| ID | Status | Decision / open question | Owner | Evidence |
|---|---|---|---|---|
| D-06 | **DECIDED (2026-09-22)** | Wall handling is **exact specular reflection with folding** (`p → 2L − p` after `floor(x/L)` crossings; normal velocity flips on an odd crossing count). Rationale from the audit: the clamp deletes the overshoot (one-step error 0.4 in the analytic case), is **not time-reversible** (max error 0.20 vs 1.78e-15 for mirror), and inflates the shipping-configuration drift by **+28 %** (clamp 2.3909296e-04 vs mirror 1.7203835e-04). Implemented in `src/physics/boundaries.py`; public signature unchanged. **Note:** this changes wall-event counts and drift values relative to `results/energy/baseline.*`, which were produced with the clamp rule and remain frozen as clamp-rule evidence. | M2 | `results/audit_2026-09-22/wall/{wall_rule_one_step.csv,wall_reversibility.csv,wall_drift_comparison.csv,wall_audit.json}` |
| D-07 | **PENDING** | Drag scheme: keep the explicit `v_half` force evaluation or move to an implicit/analytic damping factor? | M3 + M4 | `tools/drag_accuracy.py`: velocity decay vs `e^{−γt}` over a `dt` ladder, fitted error order, and the reported `γ·dt` head-room against the 3e-3 criterion. |
| D-08 | **DECIDED (2026-09-22)** | Retain **`r_cut = r₀ = 0.8`** under the current force/potential model (no shifted-force or other truncation variant). Rationale from the audit: at `r_cut = r₀` both U (jump ≈ 7.5e-18) and F (jump ≈ 1.5e-8) are effectively continuous at the cutoff, and the force vanishes there; any `r_cut > r₀` introduces an O(1) energy jump per crossing event (1.2 at r_cut=1.2, 28.8 at r_cut=2.5) that is **independent of dt** and therefore cannot be reduced by timestep refinement. `data/default_config.json` unchanged. | M2 | `results/audit_2026-09-22/cutoff/{cutoff_continuity_summary.csv,cutoff_crossing.csv,cutoff_audit.json}` |
| D-09 | **PARTIALLY DECIDED (2026-09-22)** | The G2 regime audit measured the shipping N=25, σ=8 configuration as **interaction-sparse**: `U_max/E₀ = 2.175e-03`, mean/max interacting pairs 0.63/3, 0.752 pair events per particle per unit time, 243 wall events (9.72 per particle). For the upcoming physics-accuracy experiment a **separate experiment configuration `data/configs/accuracy_interacting.json` (N=50, σ=2)** is created, which the audit measured at `U_max/E₀ = 1.272e-02` with 0.468 pair events per particle per unit time and 127 wall events. This is **not** asserted to be universally optimal — it is the accuracy-study configuration with the highest measured `U_max/E₀` among the four audited probes. The final physics-evidence regime (D-09 as a whole) is still open until the Phase 3 accuracy study runs. | M2 + M3 | `results/audit_2026-09-22/regime/{regime_summary.csv,regime_audit.json}`; `data/configs/accuracy_interacting.json` |
| D-10 | **PENDING** | Accuracy observable and protocol: reference-solution error norm plus measured order, replacing `max|ΔE|/|E₀|` as the order-defining metric. | M3 | The Phase 3 accuracy study in `results/accuracy_<date>/` (design frozen before the run). |
| D-11 | **DECIDED (2026-09-22)** | Cell-list ε bug **fixed**: the comparison is now `epsilon*epsilon < dist_sq <= cutoff_sq` (ε is a *distance* threshold, matching the reference kernels and the C backend). Audit established the exact failure window (pairs with `1e-12 < r <= 1e-6` silently dropped; |ΔF| ≈ 12.0 at r = 1e-7); regression test `tests/test_innovation.py::test_cell_list_close_pair_epsilon_convention` now covers separations 1e-13 … 1e-3. No stored artifact contains such a pair, so no existing result is affected. | M2 (+M6) | `results/audit_2026-09-22/cell_list/close_pair_epsilon_audit.csv`; `tests/test_innovation.py` |

## 3. Parameter and design choices deliberately left open (D-12 – D-18)

These must not be chosen until D-06 – D-11 are resolved and the corresponding experiment
has been run into a new output root.

| ID | Status | Open choice | Owner | Evidence required |
|---|---|---|---|---|
| D-12 | **PENDING** | Timestep `dt` | M3 | Phase 3 accuracy study |
| D-13 | **PENDING** | Integrator (semi-implicit Euler vs velocity Verlet) | M3 | Phase 3 measured order per integrator |
| D-14 | **PENDING** | Numerical precision (float32 vs float64) | M4 | Phase 4 study with a float64 round-off noise-floor control |
| D-15 | **PENDING** | Cutoff value | M2 | D-08 audit |
| D-16 | **PENDING** | Viscous drag `γ` (control vs relaxation value) | M4 | Phase 4 γ sweep plus D-07 drag accuracy |
| D-17 | **PENDING** | Force backend exposed to the simulator | M5 | Phase 5 required benchmark + parity-gated end-to-end comparison |
| D-18 | **PENDING** | Innovation and its headline metric | M6 (+M5) | Measured before/after metric (Phase 6); the current cell-list result is a negative wall-clock result and must not be presented otherwise |

## 4. Superseded evidence

| Date | Item | Reason |
|---|---|---|
| 2026-09-22 | `results/timestep/*`, `results/dissipation/*`, `results/precision/*` (all dated 17-09) | Not reproducible with the current code (`LEGACY_IC_MISMATCH`, `docs/PROVENANCE.md`). Retained in the frozen snapshot only; to be superseded by Phases 3–4. |
| 2026-09-22 | The dt and precision tables in `report/FINAL_REPORT.md` §§3–4 | Derived from the artifacts above; mixed with the current baseline in the same section. |
| 2026-09-22 | `report/figures/baseline_energy.png`, `visualisation/baseline.gif` | Predate the 25-particle baseline; regeneration is scheduled (Phase 8). |
