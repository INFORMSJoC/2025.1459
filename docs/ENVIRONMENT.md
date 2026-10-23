# Installation and numerical environment

Install Julia **1.10.4** for HPR-QP, PDQP, SCS and Gurobi, and Julia **1.11.5** for CuClarabel. Set `JULIA_BIN` and `CUCLARABEL_JULIA_BIN` to the respective executables. `--julia` and `--cuclarabel-julia` provide command-line overrides. Both setup and mixed-solver runs select the executable for each solver; a standalone CuClarabel invocation can also pass its Julia 1.11.5 executable through `--julia`. From the package root, run `./scripts/run.sh setup SOLVER`, or `./scripts/run.sh setup ALL` for all solver environments. Setup instantiates the supplied manifests and downloads missing dependencies; it does not update package versions or install Julia, GPU drivers or a Gurobi license. Linux x86_64 was tested.

| Solver | Pinned environment | Runtime / source |
|---|---|---|
| HPR-QP and its variants | `scripts/environments/` | CUDA.jl 5.11.0; CUDA 13.2; CUDSS.jl 0.6.7 |
| PDQP | `scripts/environments/PDQP/` | Collected PDQP source; CUDA.jl 5.11.0; CUDA 13.2 |
| SCS | `scripts/environments/SCS/` | SCS.jl 2.1.0, native SCS 3.2.7 GPU artifact; CUDA 11.8 |
| CuClarabel | `scripts/environments/CuClarabel/` | Original CuClarabel 0.10.0; Julia 1.11.5; CUDA.jl 5.8.2; CUDSS.jl 0.4.4; CUDA 12.8 |
| Gurobi | `scripts/environments/Gurobi/` | Gurobi.jl 1.9.2, native Gurobi 12.0.2; CPU; own valid license required |

JuMP 1.30.0 is pinned for SCS and Gurobi; CuClarabel uses JuMP 1.25.0. PDQP uses its original Julia command-line driver. Local package dependencies in the manifests resolve to `src/competitors/`. The default dependency cache is `.julia-depot`; `JULIA_DEPOT_PATH`, or separate `HPR_DEPOT_PATH` / `COMPETITOR_DEPOT_PATH`, can select existing caches. CuClarabel's Python dependency environment can initialize automatically on first use.

The HPR environment lock records hashes of the Project, Manifest and CUDA preferences. HPR workers check Julia/CUDA versions, CUDA runtime, and one Julia/BLAS thread before solving. Native result records include environment information. Julia and BLAS default to one thread; Gurobi defaults to 32 native threads, configurable with `--cpu-threads`.

## Original CuClarabel environment

The bundled CuClarabel source matches upstream commit `83b292a00a927419eeb0cb00f05f38e2119efb17`; no source patches are applied. Its environment also pins GPUCompiler 1.5.0, LLVM.jl 9.4.0, GPUArrays 11.2.2 and KernelAbstractions 0.9.34.

`LocalPreferences.toml` uses SnoopPrecompile's supported `skip_precompile = ["Clarabel"]` setting. This skips the upstream import-time sample workload, which fails when generating a CPU package image from GPU instructions. Actual solver kernels are compiled during the separate warmup solve, before the measured solve. Solver arithmetic, stopping tolerances and source files are unchanged.

This environment was tested separately on an H100 with NVIDIA driver **570.133.20** and CUDA runtime **12.8.0**. CUDA.jl 5.8.2 rejects a driver reporting CUDA 13; use a compatible driver environment. The package does not install or replace the system GPU driver.

MAT is needed for HPR-QP's implicit and trust-region inputs and for the SCS/CuClarabel/Gurobi trust-region drivers, which read the shared `.mat` instances. CuClarabel uses MAT 0.10.7 to preserve its older dependency graph; ordinary MPS inputs do not invoke this reader.

## Tested hardware

Reproduction runs used NVIDIA A100-SXM4-80GB GPUs, NVIDIA driver 580.126.16, Intel Xeon Platinum 8338C processors at 2.60 GHz, about 251 GiB host RAM and Ubuntu 22.04.3. These pinned environments provide the tested reproduction configuration. The manuscript reports Ubuntu 24.04.2; timings may vary with the operating system, hardware and concurrent workloads.

## Solver selection and settings

`HPR-QP` selects the implicit implementation for Lasso and QAP. Explicit CQP experiments use `src/variants/main/`. `HPR-QP-explicit` selects the main implementation for Lasso with `force_general_q=true`, disabling the diagonal-Q shortcut for the paper's explicit-formulation comparison. `HPR-p1` and `HPR-p2` select the primal variants. The [experiment mapping](../README.md#replicating) lists supported choices.

GPU memory, CPU preprocessing and storage requirements vary by dataset. Consult [experiment costs](EXPERIMENTS.md) before full runs. Use an idle machine for timing comparisons; the default launcher executes sequentially. GPU drivers, Julia and a valid Gurobi license must be installed separately. Source notices and licenses are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
