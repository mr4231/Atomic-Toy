# Final Scientific Computing Report: 3D Atomic Toy Model
**Course**: Fundamentals of Scientific Computing, IIT Hyderabad
**Repository**: reproducible baseline
**Evidence Source**: Raw numeric data generated under `results/`

---

## 1. Model and Equations

The system models $N$ classical particles in a 3D cubic domain $[0, L]^3$ interacting via a truncated harmonic contact potential:

$$U_{ij}(r) = \begin{cases} \frac{1}{2} k (r_{ij} - r_0)^2 & \text{if } r_{ij} \le r_{\text{cut}} \\ 0 & \text{if } r_{ij} > r_{\text{cut}} \end{cases}$$

where $r_{ij} = \|r_i - r_j\|$ is the inter-particle distance, $r_0$ is the equilibrium distance, and $k$ is the spring constant.

The conservative pair force on particle $i$ due to particle $j$ is:
$$F_i^{\text{pair}} = -k (r_{ij} - r_0) \frac{r_i - r_j}{r_{ij}} \quad (r_{ij} \le r_{\text{cut}})$$

Boundary condition: Particles overshooting the cubic box $[0, L]^3$ undergo hard-wall specular reflection where normal velocity is reversed ($v_n \to -v_n$) and positions are clamped to $[0, L]$.

---

## 2. Model Verification

1. **Distance Symmetry**: $\|r_i - r_j\| = \|r_j - r_i\|$ verified for all pairs ($\|D - D^T\| = 0$).
2. **Zero Diagonal**: $d_{ii} = 0$ for all $i \in \{1, \dots, N\}$.
3. **No Self-Neighbor**: Neighbor matrix diagonal is strictly `False`.
4. **Confinement**: 100% of particles remain strictly contained within $0 \le x, y, z \le L$ across all timesteps.
5. **Net Internal Force Balance**: $\sum F_i = 0$ (Newton's 3rd Law exact pair cancellation).

---

## 3. Accuracy and Integrator Justification

Baseline configuration max relative energy drift: **2.391e-04** over 500 steps.

### Measured Timestep Sweep (Velocity Verlet)

| dt | Steps | Integrator | Max Relative Energy Drift |
|---|---|---|---|
| 0.05 | 100 | velocity_verlet | 2.063e-03 |
| 0.02 | 250 | velocity_verlet | 3.227e-04 |
| 0.01 | 500 | velocity_verlet | 8.058e-05 |
| 0.005 | 1000 | velocity_verlet | 2.013e-05 |

**Integrator Justification**: Velocity-Verlet is selected over Semi-Implicit Euler because it is a second-order symplectic integrator ($O(\Delta t^2)$ energy error scaling). Halving the timestep from $dt=0.01$ to $dt=0.005$ drops the maximum relative energy drift by a factor of ~4 (from $8.06 \times 10^{-5}$ to $2.01 \times 10^{-5}$), confirming theoretical $O(\Delta t^2)$ convergence.

---

## 4. Dissipation and Precision Analysis

### Dissipation (Linear Viscous Drag $F_{\text{drag}} = -\gamma v$)

Relaxation criterion: Relative energy change $< 3 \times 10^{-3}$ AND mean speed $< 3 \times 10^{-2}$ over a 20-step window.

| Gamma (γ) | Steps | Final Total Energy | Final Mean Speed | Relaxation Time (s) |
|---|---|---|---|---|
| 0.0 | 500 | 0.2749 | 0.1895 | not_reached |
| 0.05 | 500 | 0.1765 | 0.1508 | not_reached |
| 0.1 | 500 | 0.1146 | 0.1205 | not_reached |
| 0.5 | 500 | 0.0061 | 0.0230 | 4.84 |

### Precision Comparison (float32 vs float64)

| Data Type (dtype) | Position Bytes | Pair Storage Bytes | Max Relative Energy Drift | Final Position Diff vs float64 |
|---|---|---|---|---|
| float32 | 144 | 1728 | 8.489e-05 | 6.518e-07 |
| float64 | 288 | 3456 | 8.058e-05 | 0.000e+00 |

---

## 5. Efficiency and Memory Scaling

### Pair-Force Kernel Benchmark (Median Wall Time over 5 Repeats)

| Implementation | N Particles | Median Time (s) | Speedup vs Python Loop | Force Parity Norm vs Reference |
|---|---|---|---|---|
| python_loop | 50 | 0.000460 | 1.00x | 0.00e+00 |
| numpy_vectorized | 50 | 0.000160 | 2.87x | 0.00e+00 |
| cell_list_spatial | 50 | 0.000918 | 0.50x | 2.25e-14 |
| c_optimized | 50 | 0.000015 | 30.66x | 2.03e-14 |
| python_loop | 200 | 0.007827 | 1.00x | 0.00e+00 |
| numpy_vectorized | 200 | 0.003954 | 1.98x | 0.00e+00 |
| cell_list_spatial | 200 | 0.009171 | 0.85x | 1.63e-13 |
| c_optimized | 200 | 0.000057 | 136.84x | 1.13e-13 |
| python_loop | 500 | 0.052201 | 1.00x | 0.00e+00 |
| numpy_vectorized | 500 | 0.027446 | 1.90x | 0.00e+00 |
| cell_list_spatial | 500 | 0.050021 | 1.04x | 6.51e-13 |
| c_optimized | 500 | 0.000334 | 156.34x | 3.72e-13 |
| python_loop | 1000 | 0.210805 | 1.00x | 0.00e+00 |
| numpy_vectorized | 1000 | 0.104855 | 2.01x | 0.00e+00 |
| cell_list_spatial | 1000 | 0.210798 | 1.00x | 2.25e-12 |
| c_optimized | 1000 | 0.001390 | 151.63x | 9.88e-13 |

### Dense Pair-Displacement Memory Storage (Theoretical Scaling)

| N Particles | Data Type | Storage (Bytes) | Storage (MiB) |
|---|---|---|---|
| 100 | float32 | 120000 | 0.11 MiB |
| 100 | float64 | 240000 | 0.23 MiB |
| 1000 | float32 | 12000000 | 11.44 MiB |
| 1000 | float64 | 24000000 | 22.89 MiB |
| 5000 | float32 | 300000000 | 286.10 MiB |
| 5000 | float64 | 600000000 | 572.20 MiB |
| 10000 | float32 | 1200000000 | 1144.41 MiB |
| 10000 | float64 | 2400000000 | 2288.82 MiB |

---

## 6. Innovation — 3D Spatial Partitioning (Cell-List)

**Innovation**: **3D Spatial Partitioning (Cell-List)**.
- **Stream**: M6 (Visualisation & Innovation) / M5 (Performance).
- **Description**: Implements a 3D spatial domain decomposition (cell-linked list) that reduces candidate pair evaluations for short-range interactions under appropriate spatial assumptions ($r_{\text{cut}} \ll L$ in a cubic domain), querying only particles in 26 adjacent neighbor cells ($d_{\text{cell}} \ge r_{\text{cut}}$) rather than all $N(N-1)/2$ pairs.
- **Correctness & Verification**: Numerical force parity verified: norm difference < 2.3 × 10^-12 against the double-loop and vectorized reference implementations across all tested configurations.
- **Measured Benchmark Performance**: The measured timings in `benchmark.csv` do not demonstrate an empirical performance improvement over the baseline Python loop at tested particle counts ($N=50: 0.50\times$, $N=200: 0.85\times$, $N=500: 1.04\times$, $N=1000: 1.00\times$) due to pure-Python cell dictionary and loop traversal overhead. The compiled C backend (`c_optimized`) remains the definitive high-performance implementation (~150× faster).

---

## 7. Final Recommendations

| Parameter | Recommended Choice | Evidence Reference |
|---|---|---|
| **Data Type (dtype)** | `float64` | `precision_summary.csv`: float32 incurs higher drift ($8.49 \times 10^{-5}$) vs float64 ($8.06 \times 10^{-5}$). |
| **Timestep (dt)** | `0.01` | `timestep_summary.csv`: Excellent stability ($8.06 \times 10^{-5}$ drift) with optimal throughput. |
| **Integrator** | `velocity_verlet` | `timestep_summary.csv`: $O(\Delta t^2)$ symplectic energy preservation. |
| **Force Backend** | `c_optimized` | `benchmark.csv`: ~150x speedup over Python loop on $N=1000$. |
| **Cutoff Radius** | `0.8` | `default_config.json`: Matches equilibrium distance $r_0=0.8$, ensuring potential continuity $U(r_{\text{cut}})=0$. |
| **Viscous Drag (γ)** | `0.0` (Control) / `0.5` (Relaxation) | `dissipation_summary.csv`: $\gamma=0.5$ reaches settled relaxation at $t=4.84$s. |
| **Spatial Kernel** | `cell_list_spatial` (Prototype) | `test_innovation.py`: Correctness verified (norm < 2.3e-12); reduces candidate pair checks theoretically, though pure-Python overhead yields no empirical speedup over reference loop. |