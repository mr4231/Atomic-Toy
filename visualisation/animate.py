"""Render or interactively view a saved 3D trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from src.utils.io import load_result

# Interactive backends need a strong reference until their window is closed.
_LIVE_ANIMATIONS: list[FuncAnimation] = []


def _draw_box(ax, length: float) -> None:
    """Draw the twelve cube edges so the simulation boundary is visible."""
    corners = np.array([[x, y, z] for x in (0, length) for y in (0, length) for z in (0, length)])
    for i, first in enumerate(corners):
        for second in corners[i + 1 :]:
            if np.sum(first != second) == 1:
                ax.plot(*zip(first, second), color="black", linewidth=0.6, alpha=0.45)


def _create_trajectory_animation(
    input_stem: str | Path, fps: int, playback_speed: float = 1.0
) -> tuple[plt.Figure, FuncAnimation]:
    """Build one reusable 3D animation from saved frames without changing the data."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if playback_speed <= 0:
        raise ValueError("playback_speed must be positive")
    result = load_result(input_stem)
    length = result.config.box_length
    figure = plt.figure(figsize=(6, 6))
    axis = figure.add_subplot(111, projection="3d")
    axis.set(xlim=(0, length), ylim=(0, length), zlim=(0, length), xlabel="x", ylabel="y", zlabel="z")
    axis.set_box_aspect((1, 1, 1))
    _draw_box(axis, length)
    scatter = axis.scatter([], [], [], s=28, c="tab:blue", depthshade=True)
    title = axis.set_title("")
    displayed_frames: list[int] = []

    def update(frame: int):
        points = result.positions[frame]
        # Use Matplotlib's public 3D collection APIs so GUI backends redraw every frame.
        scatter.set_offsets(points[:, :2])
        scatter.set_3d_properties(points[:, 2], zdir="z")
        title.set_text(f"t = {result.frame_times[frame]:.3f}")
        displayed_frames.append(frame)
        return scatter, title

    update(0)  # Populate frame 0 before the GUI event loop starts.
    animation = FuncAnimation(
        # Playback speed changes only this GUI timer; stored times and positions are untouched.
        figure, update, frames=range(len(result.positions)), interval=1000 / (fps * playback_speed),
        blit=False, cache_frame_data=False,
    )
    # Keep a reference on the figure so interactive backends do not garbage-collect the animation.
    figure._trajectory_animation = animation
    figure._trajectory_displayed_frames = displayed_frames
    return figure, animation


def animate_trajectory(input_stem: str | Path, output: str | Path, fps: int = 25) -> Path:
    """Save a GIF or MP4 directly from stored simulation frames."""
    figure, animation = _create_trajectory_animation(input_stem, fps)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".gif":
        animation.save(output_path, writer=PillowWriter(fps=fps))
    elif output_path.suffix.lower() == ".mp4":
        animation.save(output_path, fps=fps)
    else:
        raise ValueError("output must end in .gif or .mp4")
    plt.close(figure)
    return output_path


def view_trajectory(input_stem: str | Path, fps: int = 25, playback_speed: float = 1.0) -> None:
    """Open a rotatable Matplotlib 3D window and play the stored simulation frames."""
    figure, animation = _create_trajectory_animation(input_stem, fps, playback_speed)
    _LIVE_ANIMATIONS.append(animation)
    try:
        # block=True prevents an interactive session from immediately closing a visible window.
        plt.show(block=True)
    finally:
        _LIVE_ANIMATIONS.remove(animation)
        plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_stem", help="Path without .npz/.json suffix")
    parser.add_argument("output", help="GIF or MP4 path")
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()
    print(animate_trajectory(args.input_stem, args.output, args.fps))
