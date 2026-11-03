"""Shared state objects used by every simulator subsystem."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, init=False)
class SimulationConfig:
    """Parameters for one reproducible 3D simulation run.

    ``number_of_particles`` and ``box_size`` are the canonical public names.
    The legacy ``n_particles`` and ``box_length`` aliases remain available for
    older tests and previously saved result metadata.
    """

    number_of_particles: int = 12
    box_size: float = 10.0
    cutoff: float = 0.8
    spring_constant: float = 15.0
    equilibrium_distance: float = 0.8
    initial_velocity_scale: float = 2.0
    minimum_initial_separation: float = 0.5
    dt: float = 0.01
    steps: int = 500
    gamma: float = 0.0
    integrator: str = "velocity_verlet"
    dtype: str = "float64"
    seed: int = 0
    frame_stride: int = 1

    def __init__(
        self,
        number_of_particles: int | None = None,
        box_size: float | None = None,
        cutoff: float = 0.8,
        spring_constant: float = 15.0,
        equilibrium_distance: float = 0.8,
        initial_velocity_scale: float = 2.0,
        minimum_initial_separation: float = 0.5,
        dt: float = 0.01,
        steps: int = 500,
        gamma: float = 0.0,
        integrator: str = "velocity_verlet",
        dtype: str = "float64",
        seed: int = 0,
        frame_stride: int = 1,
        *,
        n_particles: int | None = None,
        box_length: float | None = None,
    ) -> None:
        """Accept canonical names and compatibility aliases without duplicate settings."""
        if number_of_particles is not None and n_particles is not None and number_of_particles != n_particles:
            raise ValueError("number_of_particles and n_particles disagree")
        if box_size is not None and box_length is not None and box_size != box_length:
            raise ValueError("box_size and box_length disagree")
        object.__setattr__(
            self, "number_of_particles",
            12 if number_of_particles is None and n_particles is None else (
                number_of_particles if number_of_particles is not None else n_particles
            ),
        )
        object.__setattr__(
            self, "box_size",
            10.0 if box_size is None and box_length is None else (
                box_size if box_size is not None else box_length
            ),
        )
        for name, value in {
            "cutoff": cutoff,
            "spring_constant": spring_constant,
            "equilibrium_distance": equilibrium_distance,
            "initial_velocity_scale": initial_velocity_scale,
            "minimum_initial_separation": minimum_initial_separation,
            "dt": dt,
            "steps": steps,
            "gamma": gamma,
            "integrator": integrator,
            "dtype": dtype,
            "seed": seed,
            "frame_stride": frame_stride,
        }.items():
            object.__setattr__(self, name, value)

    @property
    def n_particles(self) -> int:
        """Compatibility alias for existing code and historical result metadata."""
        return self.number_of_particles

    @property
    def box_length(self) -> float:
        """Compatibility alias for existing code and historical result metadata."""
        return self.box_size

    @property
    def numpy_dtype(self) -> np.dtype:
        """Return the configured floating-point type after validation."""
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")
        return np.dtype(self.dtype)

    def validate(self) -> None:
        """Fail early when a configuration cannot describe a physical run."""
        if self.number_of_particles < 1:
            raise ValueError("number_of_particles must be at least 1")
        if self.box_size <= 0 or self.cutoff <= 0 or self.dt <= 0:
            raise ValueError("box_size, cutoff, and dt must be positive")
        if self.spring_constant < 0 or self.equilibrium_distance < 0:
            raise ValueError("spring parameters cannot be negative")
        if self.initial_velocity_scale < 0 or self.minimum_initial_separation < 0:
            raise ValueError("initial velocity scale and minimum separation cannot be negative")
        if self.steps < 0 or self.frame_stride < 1 or self.gamma < 0:
            raise ValueError("steps/frame_stride/gamma are outside their valid range")
        if self.integrator not in {"semi_implicit_euler", "velocity_verlet"}:
            raise ValueError("integrator must be semi_implicit_euler or velocity_verlet")
        _ = self.numpy_dtype

    def as_dict(self) -> dict[str, Any]:
        """Provide JSON-safe metadata alongside every saved experiment."""
        return asdict(self)


@dataclass
class SimulationState:
    """Positions, velocities, and masses in the canonical (N, 3)/(N,) layout."""

    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray
    step: int = 0

    def validate(self) -> None:
        """Check shape and finite values before numerical work begins."""
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if self.velocities.shape != self.positions.shape:
            raise ValueError("velocities must have the same shape as positions")
        if self.masses.shape != (self.positions.shape[0],):
            raise ValueError("masses must have shape (N,)")
        if np.any(self.masses <= 0):
            raise ValueError("all masses must be positive")
        if not all(np.all(np.isfinite(x)) for x in (self.positions, self.velocities, self.masses)):
            raise ValueError("state contains non-finite values")

    def copy(self) -> "SimulationState":
        """Copy arrays so experiment frames never alias the live state."""
        return SimulationState(
            self.positions.copy(), self.velocities.copy(), self.masses.copy(), self.step
        )


@dataclass
class SimulationResult:
    """Saved trajectory and diagnostics returned by a completed run."""

    times: np.ndarray
    frame_times: np.ndarray
    positions: np.ndarray
    kinetic: np.ndarray
    potential: np.ndarray
    total_energy: np.ndarray
    mean_speed: np.ndarray
    config: SimulationConfig
    final_state: SimulationState = field(repr=False)

    @property
    def max_relative_energy_drift(self) -> float:
        """Measure conservative-run energy stability without dividing by zero."""
        reference = float(self.total_energy[0])
        scale = max(abs(reference), np.finfo(self.total_energy.dtype).eps)
        return float(np.max(np.abs(self.total_energy - reference) / scale))
