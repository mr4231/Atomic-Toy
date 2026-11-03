"""Generate the energy and performance figures required by the project brief."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.io import load_result


def plot_energy(input_stem: str | Path, output: str | Path) -> Path:
    """Plot K(t), U(t), and E(t) from one saved simulation result."""
    result = load_result(input_stem)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(result.times, result.kinetic, label="Kinetic K")
    axis.plot(result.times, result.potential, label="Potential U")
    axis.plot(result.times, result.total_energy, label="Total E", linewidth=2)
    axis.set(xlabel="Time", ylabel="Energy", title="Energy diagnostics")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def plot_performance(input_csv: str | Path, output: str | Path) -> Path:
    """Plot median pair-force time against N on the requested logarithmic time scale."""
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with Path(input_csv).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            series[row["implementation"]].append((int(row["n_particles"]), float(row["median_seconds"])))
    figure, axis = plt.subplots(figsize=(7, 4))
    for name, values in sorted(series.items()):
        values.sort()
        axis.plot([n for n, _ in values], [seconds for _, seconds in values], marker="o", label=name)
    axis.set_yscale("log")
    axis.set(xlabel="Particle count N", ylabel="Median wall time (s, log scale)", title="Pair-force performance")
    axis.grid(alpha=0.3, which="both")
    axis.legend()
    figure.tight_layout()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def plot_energy_comparison(
    input_stems: dict[str, str | Path], output: str | Path, *, title: str
) -> Path:
    """Plot total energy for labelled runs, preserving all raw run files."""
    figure, axis = plt.subplots(figsize=(7, 4))
    for label, stem in input_stems.items():
        result = load_result(stem)
        axis.plot(result.times, result.total_energy, label=label)
    axis.set(xlabel="Time", ylabel="Total energy", title=title)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def plot_precision_runtime(input_csv: str | Path, output: str | Path) -> Path:
    """Plot the median end-to-end float32/float64 runtime from a precision study."""
    with Path(input_csv).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(
        [row["dtype"] for row in rows],
        [float(row["execution_median_seconds"]) for row in rows],
        color=["tab:orange", "tab:blue"],
    )
    axis.set(xlabel="Floating-point type", ylabel="Median simulation time (s)", title="Precision runtime")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="kind", required=True)
    energy = subparsers.add_parser("energy")
    energy.add_argument("input_stem")
    energy.add_argument("output")
    performance = subparsers.add_parser("performance")
    performance.add_argument("input_csv")
    performance.add_argument("output")
    precision = subparsers.add_parser("precision")
    precision.add_argument("input_csv")
    precision.add_argument("output")
    args = parser.parse_args()
    if args.kind == "energy":
        print(plot_energy(args.input_stem, args.output))
    elif args.kind == "performance":
        print(plot_performance(args.input_csv, args.output))
    else:
        print(plot_precision_runtime(args.input_csv, args.output))
