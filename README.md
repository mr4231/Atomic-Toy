# 3D Atomic Toy Model

This is a reproducible baseline for the Fundamentals of Scientific Computing atomic-interaction project. It models particles in a cubic 3D box, reflects hard walls, applies cutoff harmonic pair forces, and records evidence for accuracy, dissipation, precision, performance, memory, and visual diagnostics.

## Quick start

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
gcc -std=c11 -O2 c_backend/pairforce.c tests/test.c -o tests/test.exe -lm; .\tests\test.exe
python run_demo.py --output results/energy_2026-10-25/baseline
python run_demo.py --output results/energy_2026-10-25/baseline --gif
python run_demo.py --output results/energy_2026-10-25/baseline --view
python run_demo.py --output results/energy_2026-10-25/baseline --view --speed 5
python web_server.py
```

The default run uses 25 particles and 500 steps. It writes a compressed trajectory and energy history to the output stem you provide; `results/energy/` is frozen, so use a new output root such as `results/energy_2026-10-25/baseline`.

Use `python run_demo.py --view` to open the saved frames in an interactive Matplotlib 3D window. Drag to rotate, use the normal Matplotlib controls to zoom, and close the window when finished. `--fps` sets the base display rate (default `25`), while `--speed` is an interactive playback multiplier; for example, `python run_demo.py --view --speed 5` plays the unchanged stored frames five times faster.

**Simulation timestep versus playback speed:** `dt` is the physical time between numerical integration steps and is stored in the trajectory. Changing it would change the simulation. `--speed` only shortens the Matplotlib timer interval while displaying already saved frames; it does not change `dt`, particle positions, energies, or any experimental result. `--speed` affects `--view` only; `--gif` continues to use `--fps`.

## Baseline initial conditions and short-range contact

The default configuration in `data/default_config.json` is the single place to tune the baseline behavior:

- `number_of_particles: 25` and `box_size: 10.0` define the random 3D initial placement volume.
- `minimum_initial_separation: 0.5` uses seeded rejection sampling to keep positions random while avoiding pathological starting overlaps.
- `initial_velocity_scale: 8.0` is the standard deviation of each independent initial velocity component. Velocities are drawn once at `t=0`; there is no thermal noise, random force, or later velocity randomization.
- `equilibrium_distance: 0.8` and `cutoff: 0.8` make the harmonic interaction contact-like: only pairs closer than the equilibrium distance interact, and that branch is repulsive.
- `dt: 0.01` remains the numerical timestep. Playback FPS and `--speed` affect only display timing, never the physical trajectory.

Each `run_demo.py` execution prints the initial minimum separation, average/maximum speed, and interacting-pair count, followed by a later trajectory pair count. This gives a quick diagnostic for close encounters without changing the model or its output data.

## Required project evidence

```powershell
# Run each scientific-computing study; output is saved under results/.
python -m src.simulation.experiments timestep --output-root results/timestep_2026-10-25
python -m src.simulation.experiments gamma --output-root results/gamma_2026-10-25
python -m src.simulation.experiments precision --output-root results/precision_2026-10-25
python -m src.simulation.experiments memory --output-root results/memory_2026-10-25
python -m src.simulation.experiments performance --output-root results/performance_2026-10-25

# Produce figures from saved, unchanged output.
python -m visualisation.plots energy results/energy/baseline report/figures/baseline_energy.png
python -m visualisation.plots performance results/performance/benchmark.csv report/figures/performance.png
```

The performance command attempts to build the C backend automatically. If no GCC/Clang compiler is installed, it still records Python and NumPy timings; install a C compiler and rerun it to fulfil the C comparison.

## Evidence discipline and provenance

The existing numeric evidence under `results/` is **frozen**: it is preserved byte-for-byte in
`evidence_frozen/2026-09-22/` with a SHA-256 manifest, and its classification is recorded in
`docs/PROVENANCE.md` (reproducible current evidence / legacy-unreproducible evidence / stale
derived artifacts) and `DECISION_LOG.md`. The freeze is *enforced*, not advisory:
`src/utils/evidence_guard.py` makes every writer raise `ProtectedEvidenceError` before any byte is
written if the target is inside a frozen study directory or would overwrite an existing file under
`results/`. Scratch paths outside `results/` are unaffected.

Consequences for the commands above: they must now target a **new** output root, for example

```powershell
python -m src.simulation.experiments timestep --output-root results/timestep_2026-09-22
python run_demo.py --output results/energy/baseline_2026-09-22
```

A refusal is always an explicit error message, never a silent overwrite. Every newly written result
stores a `provenance` block (git commit and dirty flag, branch, Python/NumPy versions, platform, the
initial-condition SHA-256, the configuration SHA-256, the command line and the write time), and each
benchmark CSV gains a `<stem>.meta.json` sidecar recording the timing protocol — wall-clock timings are
machine dependent, while the parity norms and inputs are reproducible. Check the classification of the
existing evidence at any time with:

```powershell
python -B -m tools.check_provenance --date 2026-09-22
```

## Folder map

```text
src/
  physics/        pair geometry, neighbors, forces, energies, walls
  integration/    semi-implicit Euler and velocity-Verlet updates
  dissipation/    relaxation criterion
  simulation/     runner plus evidence/benchmark commands
  utils/          portable NPZ/JSON result persistence and metrics
c_backend/        matching C harmonic-force kernel and C reference test
tests/            regression tests for required numerical checks
visualisation/    3D animation and reproducible plot scripts
data/             versioned input configuration
results/          generated raw experimental data, organised by study
docs/             protocol and architecture guidance
report/           figures, tables, and a report-writing template
notebooks/        exploratory work that imports, rather than duplicates, src/
```

## Model conventions

- `positions`, `velocities`, and `forces` are `(N, 3)` arrays; `masses` is `(N,)`.
- Initial positions are random but overlap-safe; initial velocities are intentionally larger and remain deterministic for a fixed seed.
- Pair displacement is `r_i - r_j`; distances are symmetric and self-neighbors are always false.
- A pair inside the cutoff has `U_ij = 0.5 k (r_ij - r0)^2`; each pair potential is counted once.
- The force on `i` is `-k(r_ij-r0)(r_i-r_j)/r_ij`; the loop and vectorized implementations must agree before timing.
- `gamma=0` is the conservative control. With drag, `F_drag = -gamma v`.

## Details of Members

1. M1
    
2. M2
    
3. M3
    
4. M4
    
5. M5
    
6. M6
    

## Roles and Responsibilities

1. **M1 – Team Leader, Integration and Architecture:** Responsible for overall project conceptualization, algorithm development for driver routines, integration of modules, debugging, and checking the accuracy and reliability of the simulation.
    
2. **M2 – 3D Physics and Force Calculations:** Responsible for designing and implementing the physical model, pairwise interaction forces, neighbor detection methods, and boundary conditions.
    
3. **M3 – Numerical Solvers and Integration:** Responsible for designing and implementing numerical solvers, studying integration methods, checking solver accuracy, and interpreting numerical results.
    
4. **M4 – Dissipation, Relaxation and Numerical Precision:** Responsible for studying dissipation and relaxation, investigating numerical precision, and developing test codes and scripts to evaluate the validity and limitations of the solvers.
    
5. **M5 – Performance and Optimization:** Responsible for performance analysis, benchmarking Python, NumPy and compiled C implementations, scalability studies, and memory analysis.
    
6. **M6 – Visualization and Diagnostics:** Responsible for 3D visualization, animation, plotting, simulation diagnostics, and coordination and evaluation of project innovations.
    

## Acknowledgment of AI Usage

The team acknowledges the use of AI tools during the development of this project.

**AI Tool Used:**

- ChatGPT and cline – Used for code development and debugging.
- Note that any file which is not .py or .c were completly made with the help of ai
    

AI-generated suggestions and code are reviewed and tested by the team. The team is responsible for verifying the correctness of the implementation, numerical results, and conclusions.

## Files Shared

The following files and materials are included as part of the project submission:

1. **README.md:** Contains the project overview, setup instructions, execution instructions, and information about the project structure.
    
2. **Video Files:** Recorded project presentation and demonstration videos explaining the implementation, working, experiments, and results.
    
3. **Source Code, Testing Code and Visualization Code:** Includes the simulation source code, testing scripts, visualization and animation code, and generated movie files. These materials accompany the README or equivalent instructions so that the code can be executed and tested.
    
4. **Project Documentation and Final PDF:** The Word or LaTeX project document and final PDF report, containing relevant time snapshots, plots, algorithms, implementation details, experimental observations, and results.