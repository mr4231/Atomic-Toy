# Project Architecture

All simulations use one state contract: positions and velocities have shape `(N, 3)`; masses have shape `(N,)`; pair displacement is `position_i - position_j`; pair forces are harmonic and only act within the cutoff.

Model conventions (D-06, D-08, D-11 in `DECISION_LOG.md`):

- **Walls** are exact specular hard walls: `reflect_walls` folds the position through `floor(x/L)` crossings (`p -> 2L - p` on an odd crossing count) and flips only the crossed normal velocity component. The rule is time-reversible and preserves speed; measured evidence is in `results/audit_2026-09-22/wall/`.
- **Cutoff** equals the equilibrium distance (`r_cut = r0 = 0.8`), so both the potential and the force are continuous at the cutoff and the interaction is purely repulsive. Evidence: `results/audit_2026-09-22/cutoff/`.
- **Cell list** uses the same `epsilon`-as-distance convention as the reference kernels (`epsilon**2 < r**2 < cutoff**2`). Evidence: `results/audit_2026-09-22/cell_list/`.
- Accuracy studies use `data/configs/accuracy_interacting.json` (N=50, sigma=2), kept separate from `data/default_config.json` (D-09).

`src/physics` is the reference model. `src/integration` consumes a force callback. `src/dissipation` supplies the relaxation diagnostic. `src/simulation` runs controlled experiments. `src/utils` saves evidence. `visualisation` reads saved data and never recomputes the trajectory. `c_backend` implements the same reference force law for parity and timing.

Keep experiment scripts and generated output in `results/`; do not alter the physics function to make a plot look better. Evidence written before 2026-09-22 was produced with the older clamped wall rule and stays frozen unchanged; cite `results/audit_2026-09-22/wall/wall_drift_comparison.csv` when comparing across the rule change.
