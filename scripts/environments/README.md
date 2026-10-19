# Pinned environments

`Project.toml` and `Manifest.toml` here define the HPR-QP Julia 1.10.4 environment. `PDQP/`, `SCS/`, `CuClarabel/`, and `Gurobi/` define separate competitor environments. CuClarabel uses Julia 1.11.5; all other solvers use Julia 1.10.4. Set `CUCLARABEL_JULIA_BIN` alongside `JULIA_BIN` for mixed-solver runs. `requirements-postprocess.txt` defines the Python plotting dependencies.

From the repository root, use `./scripts/run.sh setup HPR-QP` or replace the solver name. Relative source paths in competitor manifests resolve to the bundled sources. CuClarabel uses its original source with the compatible older dependency set and the precompilation preference documented in the environment details. Hardware/runtime details are in [docs/ENVIRONMENT.md](../../docs/ENVIRONMENT.md).
