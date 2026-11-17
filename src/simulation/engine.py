
from __future__ import annotations

import numpy as np

from src.integration import damped_velocity_verlet_step, semi_implicit_euler_step, velocity_verlet_step
from src.model import SimulationConfig, SimulationResult, SimulationState
from src.physics.energy import total_energy
from src.physics.forces import total_forces
from src.physics.geometry import pairwise_geometry


def create_random_state(config: SimulationConfig) -> SimulationState:
    config.validate()
    rng = np.random.default_rng(config.seed)
    dtype = config.numpy_dtype
    # Rejection sampling keeps positions random while preventing pathological overlaps.
    positions = np.empty((config.n_particles, 3), dtype=dtype)
    for particle in range(config.n_particles):
        for _ in range(10_000):
            candidate = rng.uniform(0.0, config.box_length, size=3).astype(dtype)
            if particle == 0 or np.all(
                np.linalg.norm(positions[:particle] - candidate, axis=1)
                >= config.minimum_initial_separation
            ):
                positions[particle] = candidate
                break
        else:
            raise RuntimeError(
                "Could not place particles with the requested minimum_initial_separation; "
                "reduce it or increase box_size."
            )
    # Every particle draws one independent vector at t=0; no random forces are added later.
    velocities = rng.normal(
        0.0, config.initial_velocity_scale, (config.n_particles, 3)
    ).astype(dtype)
    masses = np.ones(config.n_particles, dtype=dtype)
    state = SimulationState(positions, velocities, masses)
    state.validate()
    return state


def initial_state_diagnostics(state: SimulationState, cutoff: float) -> dict[str, float | int]:
    geometry = pairwise_geometry(state.positions, cutoff)
    unique_distances = geometry.distance[np.triu_indices(len(state.positions), k=1)]
    speeds = np.linalg.norm(state.velocities, axis=1)
    return {
        "minimum_initial_particle_separation": float(np.min(unique_distances)) if len(unique_distances) else 0.0,
        "average_initial_speed": float(np.mean(speeds)),
        "maximum_initial_speed": float(np.max(speeds)),
        "initial_interacting_pair_count": int(np.count_nonzero(np.triu(geometry.neighbors, k=1))),
    }


def _select_stepper(config: SimulationConfig):
    return velocity_verlet_step if config.integrator == "velocity_verlet" else semi_implicit_euler_step


def run_simulation(config: SimulationConfig, state: SimulationState | None = None) -> SimulationResult:
    config.validate()
    current = create_random_state(config) if state is None else state.copy()
    current.validate()
    if current.positions.shape[0] != config.n_particles:
        raise ValueError("state particle count must match config.n_particles")

    def pair_force_function(positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        return total_forces(
            positions, velocities, config.cutoff, config.spring_constant,
            config.equilibrium_distance, 0.0,
        )

    def force_function(positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        return total_forces(
            positions, velocities, config.cutoff, config.spring_constant,
            config.equilibrium_distance, config.gamma,
        )

    stepper = _select_stepper(config)
    stored_positions: list[np.ndarray] = [current.positions.copy()]
    frame_times: list[float] = [0.0]
    kinetic: list[float] = []
    potential: list[float] = []
    total: list[float] = []
    mean_speed: list[float] = []

    def record_diagnostics() -> None:
        k, u, e = total_energy(
            current.positions, current.velocities, current.masses, config.cutoff,
            config.spring_constant, config.equilibrium_distance,
        )
        kinetic.append(k)
        potential.append(u)
        total.append(e)
        mean_speed.append(float(np.mean(np.linalg.norm(current.velocities, axis=1))))

    record_diagnostics()
    for _ in range(config.steps):
        if config.integrator == "velocity_verlet" and config.gamma > 0:
            current = damped_velocity_verlet_step(
                current, config.dt, config.box_length, pair_force_function, config.gamma
            )
        else:
            current = stepper(current, config.dt, config.box_length, force_function)
        record_diagnostics()
        if current.step % config.frame_stride == 0 or current.step == config.steps:
            stored_positions.append(current.positions.copy())
            frame_times.append(current.step * config.dt)

    return SimulationResult(
        times=np.arange(config.steps + 1, dtype=config.numpy_dtype) * config.dt,
        frame_times=np.asarray(frame_times, dtype=config.numpy_dtype),
        positions=np.asarray(stored_positions, dtype=config.numpy_dtype),
        kinetic=np.asarray(kinetic, dtype=config.numpy_dtype),
        potential=np.asarray(potential, dtype=config.numpy_dtype),
        total_energy=np.asarray(total, dtype=config.numpy_dtype),
        mean_speed=np.asarray(mean_speed, dtype=config.numpy_dtype),
        config=config,
        final_state=current,
    )
