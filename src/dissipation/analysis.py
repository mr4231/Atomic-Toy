from __future__ import annotations

import numpy as np


def relaxation_time(
    times: np.ndarray,
    total_energy: np.ndarray,
    mean_speed: np.ndarray,
    *,
    energy_relative_change: float = 3e-3,
    speed_threshold: float = 3e-2,
    window: int = 20,
) -> float | None:
    """return the first settled time, or none if no final-window criterion is met."""
    if len(times) != len(total_energy) or len(times) != len(mean_speed):
        raise ValueError("times, total_energy, and mean_speed must have equal length")
    if window < 2 or len(times) < window:
        return None

    scale = max(abs(float(total_energy[0])), np.finfo(total_energy.dtype).eps)
    for end in range(window - 1, len(times)):
        start = end - window + 1
        energy_change = abs(total_energy[end] - total_energy[start]) / scale
        average_speed = float(np.mean(mean_speed[start : end + 1]))
        # Both conditions prevent a conservative but still-moving system from being mislabelled settled.
        if energy_change < energy_relative_change and average_speed < speed_threshold:
            return float(times[end])
    return None
