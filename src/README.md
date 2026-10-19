# Solver sources

- [variants](variants/README.md): six author-developed implementations used only for this project, covered by the repository MIT license.
- `competitors/CuClarabel`, `competitors/PDQP`, `competitors/SCS`: preserved third-party snapshots, each with its own LICENSE. Gurobi is an external licensed dependency.
- `competitors/patches`: changes to third-party snapshots used by the reproduction workflow.

Use `../scripts/run.sh` from this directory, or `./scripts/run.sh` from the repository root. Do not run historical drivers directly for the paper workflow.
