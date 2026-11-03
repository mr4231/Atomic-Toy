# Frozen evidence snapshot

This directory is an immutable copy of the project's experiment evidence, taken
before any repair work started.

* `results/`      - verbatim copy of the live `results/` tree
* `report/`       - verbatim copy of the report tree and figures
* `baseline.gif`  - the committed animation artifact
* `MANIFEST.csv`  - size, modification time and SHA-256 of every frozen file
* `GIT_STATE.txt` - revision, dirtiness, untracked hashes, and the pre-commit patch
* `working_tree.patch` - the uncommitted diff at freeze time

Rules for everything that follows:

1. Nothing in this directory may be modified, regenerated or deleted.
2. New runs write to a **new output root** (for example
   `python -m src.simulation.experiments timestep --output-root results/timestep_s2`)
   and never into `results/`.
3. `data/default_config.json` is frozen as-is; new studies get their own
   versioned copy under `data/configs/`.
4. Any file listed in `MANIFEST.csv` with `provenance_verdict` =
   `LEGACY_IC_MISMATCH` is not reproducible with the current code and must not be
   quoted elsewhere as current evidence. See `docs/PROVENANCE.md`.
5. Do not check out another branch while `main` holds the only copy of this
   work: `git checkout main` from a freeze branch would remove the files that
   exist only on that branch.
