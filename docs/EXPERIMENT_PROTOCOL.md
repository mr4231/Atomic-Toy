# Experiment Protocol for team members from me(Anish)

Everyone follow these...
Use the same seed, initial state, box, cutoff, spring parameters, and physical horizon for a comparison. Change only the stated variable.

- Timestep: test `0.05`, `0.02`, `0.01`, and `0.005`; report maximum relative total-energy drift.
- Dissipation: test gamma `0`, `0.05`, `0.1`, and `0.5` over the common configured horizon; report K, U, E, mean speed, and the stored relaxation criterion. A run is called relaxed only when both the relative energy change is below `3e-3` and the mean speed is below `3e-2` over a 20-step window. The two thresholds are explicit study choices, not universal constants.
- Precision: compare float32 and float64 from identical initial arrays; report array bytes, drift, and normalized final-position difference.
- Performance: use N `50`, `200`, `500`, `1000`; perform at least five repetitions; report median kernel time, speedup, and force parity before timing.
- Memory: calculate the `(N, N, 3)` array size without allocating unsafe arrays.

Record Git commit and machine details in the final report alongside each performance table.
