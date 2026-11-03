"""Gate G2.4 (D-11): establish the cell-list close-pair epsilon discrepancy.

Measurement only.  ``src/physics/cell_list.py`` is **not** modified.  This audit
compares the production reference kernels against the cell list for particle
separations from 1e-12 to 1e-5 (far below any physically initialised separation,
whose minimum is 0.5) and records exactly where the two epsilon conventions
disagree.

Conventions under test:
- reference (``geometry.py`` / ``forces.py``): ``distance > 1e-12``, i.e. the
  *distance* (equivalently ``dist_sq > 1e-24``) is compared against the threshold;
- cell list (``cell_list.py:92``): ``epsilon < dist_sq``, i.e. the *squared
  distance* is compared against the same unsquared 1e-12, which excludes pairs
  with ``r <= 1e-6``.

Outputs: ``results/audit_<date>/cell_list/``

Run::

    python -B tools/cell_list_audit.py --date 2026-09-22
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit_common import (  # noqa: E402
    audit_output_dir, load_shipping_config, provenance, write_csv, write_json,
)

from src.physics.cell_list import pair_forces_cell_list  # noqa: E402
from src.physics.forces import pair_forces_loop, pair_forces_vectorized  # noqa: E402

SEPARATIONS = tuple(10.0 ** exponent for exponent in (-12, -11, -10, -9, -8, -7, -6, -5))
BOX = 10.0  # only used to place the pair; walls play no role in a static probe


def pair_forces_at(separation: float, shipping) -> dict:
    """Compare the three kernels for one pair at the given separation."""
    positions = np.array([[BOX / 2, BOX / 2, BOX / 2],
                          [BOX / 2 + separation, BOX / 2, BOX / 2]])
    loop = pair_forces_loop(positions, shipping.cutoff, shipping.spring_constant,
                            shipping.equilibrium_distance)
    vectorized = pair_forces_vectorized(positions, shipping.cutoff,
                                        shipping.spring_constant,
                                        shipping.equilibrium_distance)
    cells = pair_forces_cell_list(positions, BOX, shipping.cutoff,
                                  shipping.spring_constant,
                                  shipping.equilibrium_distance)
    reference = loop[0]          # force on the first particle
    cell = cells[0]
    return {
        "separation": separation,
        "reference_force_x": float(reference[0]),
        "reference_force_magnitude": float(np.linalg.norm(reference)),
        "cell_list_force_x": float(cell[0]),
        "cell_list_force_magnitude": float(np.linalg.norm(cell)),
        "vectorized_matches_loop": bool(np.array_equal(loop, vectorized)),
        "cell_list_matches_reference": bool(np.array_equal(loop, cells)),
        "max_abs_difference": float(np.max(np.abs(loop - cells))),
        "reference_excludes_pair": bool(np.linalg.norm(reference) == 0.0),
        "cell_list_excludes_pair": bool(np.linalg.norm(cell) == 0.0),
        "reference_exclusion_threshold_distance": 1e-12,
        "cell_list_effective_exclusion_threshold_distance": 1e-6,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, help="audit date folder (default: today)")
    args = parser.parse_args()
    output = audit_output_dir("cell_list", args.date)

    shipping = load_shipping_config()
    rows = [pair_forces_at(separation, shipping) for separation in SEPARATIONS]

    # Control: a well-separated configuration must still agree.
    rng = np.random.default_rng(0)
    control_positions = rng.uniform(0.0, BOX, (25, 3))
    control_loop = pair_forces_loop(control_positions, shipping.cutoff,
                                    shipping.spring_constant,
                                    shipping.equilibrium_distance)
    control_cells = pair_forces_cell_list(control_positions, BOX, shipping.cutoff,
                                          shipping.spring_constant,
                                          shipping.equilibrium_distance)
    control_row = {
        "case": "25 random particles, seed 0 (well separated)",
        "max_abs_difference": float(np.max(np.abs(control_loop - control_cells))),
        "matches_reference": bool(np.array_equal(control_loop, control_cells)),
    }

    disagreements = [row["separation"] for row in rows if not row["cell_list_matches_reference"]]
    write_csv(output / "cell_list_close_pair.csv", rows)
    write_json(output / "cell_list_audit.json", {
        "audit": "G2.4 cell-list close-pair epsilon convention (D-11)",
        "shipping_config": shipping.as_dict(),
        "separations_tested": list(SEPARATIONS),
        "reference_convention": "distance > 1e-12 (geometry.py/forces.py)",
        "cell_list_convention": "epsilon(=1e-12) < dist_sq (cell_list.py:92)",
        "close_pair_rows": rows,
        "control_row": control_row,
        "disagreement_separations": disagreements,
        "provenance": provenance(config=shipping, extra={"audit": "cell_list"}),
        "note": ("measurement only; cell_list.py is unmodified and no fix is applied "
                 "or chosen here"),
    })

    print(f"audit root: {output}")
    print(f"control: max abs diff {control_row['max_abs_difference']:.3e}, "
          f"matches reference: {control_row['matches_reference']}")
    for row in rows:
        print(f"  r={row['separation']:.0e}  ref|F|={row['reference_force_magnitude']:.6e}  "
              f"cell|F|={row['cell_list_force_magnitude']:.6e}  "
              f"agree={row['cell_list_matches_reference']}  "
              f"ref_excluded={row['reference_excludes_pair']} "
              f"cell_excluded={row['cell_list_excludes_pair']}")
    print("disagreement separations:", [f"{value:.0e}" for value in disagreements])


if __name__ == "__main__":
    main()
