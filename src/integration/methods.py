"""Baseline Euler and smoother velocity-Verlet integration methods."""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.model import SimulationState
from src.physics.boundaries import reflect_walls

ForceFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


def semi_implicit_euler_step(
    state: SimulationState, dt: float, box_length: float, force_function: ForceFunction
) -> SimulationState:
    """Advance one step by updating velocity before position (the handout baseline)."""
    force = force_function(state.positions, state.velocities)
    acceleration = force / state.masses[:, None]
    velocities = state.velocities + acceleration * dt
    positions = state.positions + velocities * dt
    reflect_walls(positions, velocities, box_length)
    return SimulationState(positions, velocities, state.masses.copy(), state.step + 1)


def velocity_verlet_step(
    state: SimulationState, dt: float, box_length: float, force_function: ForceFunction
) -> SimulationState:
    """Advance one step with two half velocity updates for better energy behavior."""
    force_old = force_function(state.positions, state.velocities)
    acceleration_old = force_old / state.masses[:, None]
    half_velocity = state.velocities + 0.5 * acceleration_old * dt
    positions = state.positions + half_velocity * dt
    reflect_walls(positions, half_velocity, box_length)
    force_new = force_function(positions, half_velocity)
    acceleration_new = force_new / state.masses[:, None]
    velocities = half_velocity + 0.5 * acceleration_new * dt
    return SimulationState(positions, velocities, state.masses.copy(), state.step + 1)


def damped_velocity_verlet_step(
    state: SimulationState,
    dt: float,
    box_length: float,
    pair_force_function: ForceFunction,
    gamma: float,
) -> SimulationState:
    """Advance a linearly damped system with an exact drag/Verlet/drag split.

    Velocity Verlet is designed for forces that depend on position.  Applying
    ``-gamma * v`` inside its ordinary force callback makes the damping update
    only an approximation.  For linear drag each component has the exact flow
    ``v <- exp(-gamma * dt / m) * v``.  Two half drag steps around a
    conservative Verlet update retain that exact free-particle decay while
    retaining the existing conservative integrator for the pair force.
    """
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    if gamma == 0:
        return velocity_verlet_step(state, dt, box_length, pair_force_function)

    drag_half = np.exp(-gamma * dt / (2.0 * state.masses))[:, None]
    damped = SimulationState(
        state.positions.copy(), state.velocities * drag_half, state.masses.copy(), state.step
    )
    conservative = velocity_verlet_step(damped, dt, box_length, pair_force_function)
    return SimulationState(
        conservative.positions,
        conservative.velocities * drag_half,
        conservative.masses,
        conservative.step,
    )
