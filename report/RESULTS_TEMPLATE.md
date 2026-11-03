# Results Template

Use the generated CSV/NPZ files to fill the report. Do not replace measured values with examples.

1. Model and equations: state the 3D cutoff harmonic potential and wall convention.
2. Verification: include distance symmetry, zero diagonal, no self-neighbor, confinement, and net-force values.
3. Accuracy: show K, U, E and the timestep-drift table; justify the integrator and timestep.
4. Dissipation and precision: compare gamma cases and float32/float64 using the saved summary tables.
5. Efficiency and memory: include Python/NumPy/C timing, speedup, parity, and pair-array memory scaling.
6. Innovation: document one before/after experiment and its metric.
7. Recommendation: state the final dtype, timestep, integrator, force backend, cutoff, gamma, and innovation decision with evidence references.
