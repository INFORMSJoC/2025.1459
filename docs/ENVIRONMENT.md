# Installation and numerical environment

Install Julia **1.10.4** and set `JULIA_BIN` to its executable. From the package root, run `./scripts/run.sh setup SOLVER`, or `./scripts/run.sh setup ALL` for all solver environments. Setup instantiates the supplied manifests and downloads missing dependencies; it does not update package versions or install Julia, GPU drivers or a Gurobi license. Linux x86_64 was tested.

| Solver | Pinned environment | Runtime / source |
|---|---|---|
| HPR-QP and its variants | `scripts/environments/` | CUDA.jl 5.11.0; CUDA 13.2; CUDSS.jl 0.6.7 |
| PDQP | `scripts/environments/PDQP/` | Collected PDQP source; CUDA.jl 5.11.0; CUDA 13.2 |
| SCS | `scripts/environments/SCS/` | SCS.jl 2.1.0, native SCS 3.2.7 GPU artifact; CUDA 11.8 |
| CuClarabel | `scripts/environments/CuClarabel/` | CuClarabel 0.10.0; CUDA.jl 5.11.0; CUDSS.jl 0.6.7; CUDA 13.2 |
| Gurobi | `scripts/environments/Gurobi/` | Gurobi.jl 1.9.2, native Gurobi 12.0.2; CPU; own valid license required |

JuMP 1.30.0 is pinned for SCS, CuClarabel and Gurobi. PDQP uses its original Julia command-line driver. Local package dependencies in the manifests resolve to `src/competitors/`. The default dependency cache is `.julia-depot`; `JULIA_DEPOT_PATH`, or separate `HPR_DEPOT_PATH` / `COMPETITOR_DEPOT_PATH`, can select existing caches. CuClarabel's Python dependency environment can initialize automatically on first use.

The HPR environment lock records hashes of the Project, Manifest and CUDA preferences. HPR workers check Julia/CUDA versions, CUDA runtime, and one Julia/BLAS thread before solving. Native result records include environment information. Julia and BLAS default to one thread; Gurobi defaults to 32 native threads, configurable with `--cpu-threads`.

## Tested hardware

Reproduction runs used NVIDIA A100-SXM4-80GB GPUs, NVIDIA driver 580.126.16, Intel Xeon Platinum 8338C processors at 2.60 GHz, about 251 GiB host RAM and Ubuntu 22.04.3. These pinned environments provide the tested reproduction configuration. The manuscript reports Ubuntu 24.04.2; timings may vary with the operating system, hardware and concurrent workloads.

## Solver selection and settings

`HPR-QP` selects the implicit implementation for Lasso and QAP. Explicit CQP experiments use `src/variants/main/`. `HPR-QP-explicit` selects the main implementation for Lasso with `force_general_q=true`, disabling the diagonal-Q shortcut for the paper's explicit-formulation comparison. `HPR-p1` and `HPR-p2` select the primal variants. The [experiment mapping](../README.md#experiment-names-and-solvers) lists supported choices.

GPU memory, CPU preprocessing and storage requirements vary by dataset. Consult [experiment costs](EXPERIMENTS.md) before full runs. Use an idle machine for timing comparisons; the default launcher executes sequentially. GPU drivers, Julia and a valid Gurobi license must be installed separately. Source notices and licenses are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
