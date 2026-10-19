# Reproduction scripts

Run these commands from the repository root:

- `./scripts/run.sh list`: experiment/solver mapping.
- `./scripts/run.sh check`: release integrity, scope, layout and documentation checks; no benchmark inputs or GPU needed.
- `./scripts/run.sh setup SOLVER`: instantiate the pinned Julia environment.
- `./scripts/run.sh EXPERIMENT SOLVER --data-root /path/to/data`: solve selected experiments.
- `./scripts/run.sh refresh`, `tables`, `figures`: refresh summaries or regenerate paper exports.
- `python3 scripts/manage_data.py extract --cache /path/to/2025.1459 --data-root /path/to/data`: unpack and verify selected prepared inputs.
- `python3 scripts/test_reviewer_runner.py`: orchestration regression tests, using synthetic fixtures and simulated Julia workers.
- `python3 scripts/build_release.py --output /path/outside/repository/review.zip`: rebuild release checksums and ZIP.

[config](config/README.md) contains execution settings and data inventories, not numerical result records. [environments](environments/README.md) contains pinned dependencies. `catalog.py` joins settings to the per-solver paper CSVs under `results/paper_results`.

The launcher accepts the new per-family input layout and the previous external `QP_data` layout. Output roots now contain `reproduced_results/` and `paper_results/`. The current defaults are described in [data](../data/README.md) and [results](../results/README.md).
