# Pinned environments

`Project.toml` and `Manifest.toml` here define the HPR-QP Julia 1.10.4 environment. `PDQP/`, `SCS/`, `CuClarabel/`, and `Gurobi/` define separate competitor environments. `requirements-postprocess.txt` defines the Python plotting dependencies.

From the repository root, use `./scripts/run.sh setup HPR-QP` or replace the solver name. Relative source paths in competitor manifests have been adjusted for this directory's location; dependency versions are unchanged. Hardware/runtime details are in [docs/ENVIRONMENT.md](../../docs/ENVIRONMENT.md).
