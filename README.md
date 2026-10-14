# HPR-QP: Reproducibility package

Kaihuang Chen, Defeng Sun, Yancheng Yuan, Guojun Zhang and Xinyuan Zhao.
**HPR-QP: A dual Halpern Peaceman–Rachford method for solving large scale convex composite quadratic programming.**

This archive is distributed in association with the [INFORMS Journal on Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE) for the authors' code. The authors' rights in the [specified generated datasets](data/licenses/AUTHOR-DATA-LICENSE.txt) are licensed under CC BY 4.0. Third-party software and datasets retain their own terms. It implements the dual Halpern Peaceman–Rachford method for large-scale convex composite quadratic programming and contains the tested reproduction snapshot accompanying the paper, with test records in [validation](docs/VALIDATION.md) and installation details in [environment details](docs/ENVIRONMENT.md).

The main entry point is **`./run.sh EXPERIMENT SOLVER`**. Choose `EXP_NAME` and `SOLVER` from the mapping below. The [detailed experiment guide](docs/EXPERIMENTS.md) gives whole-dataset recommendations, input sizes and time estimates.

**Download experiment data:** [Dropbox — 2025.1459](https://www.dropbox.com/scl/fo/y0n7luz0yvc192nsk0nwa/ABllkwQ2ZeWGe9HiR6EyJpk?rlkey=y3n3na48lb9f78kpki342dw57&dl=0). Large datasets are stored here, organized by experiment. See [folder mapping and extraction instructions](data/README.md#dropbox-folder-layout). The small quick-start inputs are included in this package.

## Cite

Please cite both the paper and this repository using their respective DOIs:

- Paper: [10.1287/ijoc.2025.1459](https://doi.org/10.1287/ijoc.2025.1459)
- Repository: [10.1287/ijoc.2025.1459.cd](https://doi.org/10.1287/ijoc.2025.1459.cd)

These identifiers are assigned to repository **2025.1459**. The DOI links will become available through the journal publication process; during review, use the [repository](https://github.com/INFORMSJoC/2025.1459).

Below is the BibTeX for citing this repository snapshot:

```bibtex
@misc{HPRQP2025_1459,
  author = {Chen, Kaihuang and Sun, Defeng and Yuan, Yancheng and Zhang, Guojun and Zhao, Xinyuan},
  publisher = {INFORMS Journal on Computing},
  title = {{HPR-QP: A dual Halpern Peaceman--Rachford method for solving large scale convex composite quadratic programming}},
  year = {2026},
  doi = {10.1287/ijoc.2025.1459.cd},
  url = {https://github.com/INFORMSJoC/2025.1459},
  note = {Available for download at https://github.com/INFORMSJoC/2025.1459}
}
```

The ongoing development repository is [PolyU-IOR/HPR-QP](https://github.com/PolyU-IOR/HPR-QP). This repository contains the reproduction snapshot accompanying the paper.

## Start here

```bash
export JULIA_BIN=/path/to/julia-1.10.4/bin/julia
./run.sh check                 # Integrity, scope and bundled input checks; no GPU
./run.sh setup HPR-QP          # Install dependencies from the pinned manifest
./run.sh QAPLIB HPR-QP --plan   # Preview the complete 36-case dataset
./run.sh QAPLIB HPR-QP         # Recommended complete dataset; allow about 40–90 min
```

QAPLIB's compact HPR-QP inputs are all bundled. Other recommended complete experiments are `Restart-Sigma HPR-QP` and, after downloading the MM archive, `Primal-Dual HPR-QP`; UCI Lasso is an optional recommendation when its external data and memory are available. Recommendations are experiment/dataset-wise, not hand-picked easy cases. They do not imply that competing solvers are inexpensive on the same dataset.

Optional **instance-wise** checks remain available:

```bash
./run.sh MM HPR-QP --cases CVXQP3_S.mps QSCSD1.mps
./run.sh setup CuClarabel
./run.sh MM CuClarabel --cases CVXQP3_S.mps QSCSD1.mps
```

`./run.sh list` shows names and solvers; `./run.sh MM HPR-QP --help` lists options. `--plan` lists selected cases and missing inputs without starting Julia. `--data-root /path/to/QP_data` supplies external inputs; bundled inputs remain a fallback. `run.sh` works from another directory via its absolute path.

## Experiment names and solvers

Use the exact arguments shown in this table:

```bash
./run.sh EXP_NAME SOLVER
```

The mapping below follows the main-text table order and explicit supplement references in the revised manuscript `HPR-QP-IJOC-R2.tex`, submitted separately. Case counts are for the complete experiment per solver. 

| `EXP_NAME` | Paper experiment / reference | Cases per solver | Supported `SOLVER` arguments |
|---|---|---:|---|
| `Primal-Dual` | R2 Table 1; Online Supplement Table S1 | 30 | `HPR-QP`, `HPR-p1`, `HPR-p2` |
| `MM` | R2 Table 2; Online Supplement Appendix D.2 (performance profiles) | 137 | `HPR-QP`, `PDQP`, `SCS`, `CuClarabel`, `Gurobi` |
| `Synthetic-CQP` | R2 Table 3; Online Supplement Table S2 and Appendix D.2 (performance profiles) | 60 | `HPR-QP`, `PDQP`, `SCS`, `CuClarabel`, `Gurobi` |
| `Lasso-UCI` | R2 Table 4; Online Supplement Tables S3 (dimensions) and S4 (results) | 11 | `HPR-QP`, `HPR-QP-explicit`, `PDQP`, `SCS`, `CuClarabel`, `Gurobi` |
| `Lasso-Synthetic` | R2 Table 5; Online Supplement Table S5 | 12 | `HPR-QP`, `HPR-QP-explicit`, `PDQP`, `SCS`, `CuClarabel`, `Gurobi` |
| `QAPLIB` | R2 Table 6; Online Supplement Appendix D.2 (performance profiles) | 36 | `HPR-QP`, `PDQP`, `SCS`, `CuClarabel`, `Gurobi` |
| `Large-QAP` | R2 Table 7 | 6 | `HPR-QP` |
| `Trust-Region` | R2 Table 8; Online Supplement Figure S4 | 17 | `HPR-QP`, `SCS`, `CuClarabel`, `Gurobi` |
| `Restart-Sigma` | R2 Figure 1 | 3 curves | `HPR-QP` |
| `Appendix-Example` | Online Supplement Appendix E (as referenced by R2) | 1 | `HPR-QP`, `HPR-p1` |

`HPR-QP` uses the **implicit implementation by default for both Lasso and QAP**. Standard explicit CQPs use `src/variants/main/`. `HPR-p1` and `HPR-p2` select the two primal variants. `HPR-QP-explicit` selects the **main implementation** for the explicit CQP formulation of Lasso, with `force_general_q=true` to preserve the paper's disabled diagonal-Q shortcut. QAP experiments currently expose the implicit HPR-QP method; an explicit-HPR QAP comparison is not a manuscript target. PDQP, SCS and CuClarabel use GPUs; Gurobi uses the CPU.

`ALL` as the solver runs every solver listed for that experiment; for example, `./run.sh Primal-Dual ALL` runs all three HPR variants. `ALL` as the experiment runs every applicable experiment for the selected solver. `./run.sh ALL HPR-QP` selects the HPR-QP column only; `./run.sh ALL ALL` also includes the primal/explicit variants and competitors. Full runs can take days to weeks.

**QAPLIB data selection:** the Dropbox subfolders `QAPLIB/HPR-QP/` (compact MAT inputs for the implicit formulation) and `QAPLIB/Competitors/` (explicit MPS inputs for PDQP, SCS, CuClarabel and Gurobi) cover the same 36 benchmark cases. `run.sh` selects the format from the solver argument. All HPR-QP QAPLIB inputs are bundled; full competitor runs need the eight external MPS archive parts. These are input-format folders, not separate experiments or solver implementations. Trust-Region and Synthetic-CQP each use one Dropbox folder for all their cases. See [data folders and extraction](data/README.md#dropbox-folder-layout).

The table lists complete datasets/experiments. To select individual cases, append `--cases`, for example `./run.sh MM HPR-QP --cases CVXQP3_S.mps QSCSD1.mps`.

## Main reviewer result: each experiment's summary

**Open `experiments/<name>/summary.html`**, for example [QAPLIB](experiments/QAPLIB/summary.html), [Primal–Dual](experiments/Primal-Dual/summary.html), or [UCI Lasso](experiments/Lasso-UCI/summary.html). These are the main paper-aligned numerical results. The corresponding `summary.csv` contains solved counts and SGM10; `results.csv` contains per-instance values; `comparison.csv` compares archived and fresh values.

```text
experiments/QAPLIB/
  summary.html, summary.csv      Main experiment result, with source counts
  results.csv, comparison.csv    Per-instance values and reproduction differences
  HPR-QP/
    archived.csv                Archived paper reference values
    fresh.csv                   Latest available fresh measurements
    raw/verification/           Packaged native JSON/log evidence
    runs/<attempt>/             New runs: native data, logs, settings and record.json
  PDQP/ ... SCS/ ... CuClarabel/ ... Gurobi/ ...
```

Each available fresh result replaces its corresponding archived paper result in the experiment summary. Unrun solver/case/tolerance rows use the archived paper values. **Mixed summaries are explicitly labeled and are not full fresh benchmarks.** Each row identifies its source, measurement origin and raw record. Unsuccessful fresh solves remain visible. External guard stops/interruption/execution failures withhold the effective timing aggregate. Setup/input errors are shown as attempt errors when no numerical row exists.

The initial fresh tables include packaged server verification evidence (`measurement_origin=verification_run`), indexed in `config/verification_snapshot.csv`; new reviewer runs use `local_run`. Original native records are stored under `experiments/<name>/<solver>/raw/verification/`; shared MM/Primal-Dual records are stored once and referenced by both summaries. `raw_file` paths for packaged evidence are relative to the package root. Each run updates its experiment summary automatically; `./run.sh refresh` rebuilds all summaries without running solvers. Generated experiment CSV/HTML views are writable; the packaged native evidence under `raw/` is checksummed; the release ZIP has its own SHA-256 checksum. Existing packaged evidence does not cause a requested reviewer run to be skipped.

The current snapshot contains completed measurements for **365/365 HPR benchmark/curve checks**, including time-limit outcomes. Competitors have sampled verification across the applicable datasets. Iterations and timings can differ between runs; per-instance comparisons and raw records are included. See [validation](docs/VALIDATION.md).

## Full runs and runtime controls

```bash
./run.sh ALL HPR-QP --data-root /path/to/QP_data --plan
./run.sh ALL ALL --data-root /path/to/QP_data
# Optional parallel workers on otherwise unused GPUs:
./run.sh MM ALL --gpus 1,2,3 --jobs 3
```

**ALL can take days to weeks**; large QAP alone historically took about 35 hours for its largest case. Competing solvers may time out or exhaust memory. The launcher applies external hard wall limits, including initialization, loading and warmup. See [budget and device details](docs/EXPERIMENTS.md#long-runs-interruption-and-devices). `--hard-time-limit`, `--run-time-limit`, and `--memory-gb` provide explicit budgets. Ctrl-C stops active workers and retains evidence.

Default runs are sequential. Each selected GPU is exclusive within this package; Gurobi CPU workers are serialized. Other CPU workloads and GPU preprocessing still affect timings. Use an idle machine for performance comparisons. Reuse the same `--output` directory to resume matching completed runs, or use `--rerun` to repeat them. Native timeout/failure is a completed solver outcome, distinct from a launcher/guard failure.

## Environments and data

Requirements: Linux, Bash, Python 3.11+, Julia **1.10.4**, and a compatible NVIDIA GPU/driver for GPU solvers. Gurobi requires the reviewer's own license. `./run.sh setup SOLVER` installs the pinned Julia dependencies when missing; `setup ALL` prepares all five environments. Julia, GPU drivers and a Gurobi license are not installed automatically. Internet access is required for missing packages/artifacts. Do not update the manifests.

HPR-QP uses CUDA.jl **5.11.0**, CUDA runtime **13.2.0** and CUDSS.jl **0.6.7**. The server has A100 80 GB GPUs and about 251 GiB host RAM. Competitors have separate environments: PDQP/CuClarabel use CUDA 13.2, SCS uses its tested CUDA 11.8 artifact, and Gurobi uses native 12.0.2. Pinned dependencies and the tested hardware are listed in [environment details](docs/ENVIRONMENT.md).

The unified launcher defaults to `.julia-depot`; set `JULIA_DEPOT_PATH` to reuse another depot, or `HPR_DEPOT_PATH` / `COMPETITOR_DEPOT_PATH` for separate caches. `PYTHON_BIN` selects Python. Julia/BLAS default to one thread; Gurobi's native thread budget defaults to 32 and can be set with `--cpu-threads`.

The HPR benchmark inventory contains **279 inputs, 476.7 GB**. The package includes **40 compact inputs (7.36 MB)**: all 36 QAPLIB MAT inputs and four MM examples. Download other inputs from the [Dropbox data folder](https://www.dropbox.com/scl/fo/y0n7luz0yvc192nsk0nwa/ABllkwQ2ZeWGe9HiR6EyJpk?rlkey=y3n3na48lb9f78kpki342dw57&dl=0). Full QAPLIB competitor runs additionally require 36 explicit MPS inputs (53.4 GB). Follow the [dataset-specific download and extraction instructions](data/README.md#dropbox-folder-layout); keep all parts of each selected archive together. Dataset sources and terms are given in [data attribution](data/README.md#dataset-attribution).

For a complete data installation, download all experiment folders, then run:

```bash
python3 scripts/manage_data.py extract --include-competitors --cache /path/to/2025.1459 --data-root /path/to/QP_data
python3 scripts/manage_data.py verify --include-competitors --data-root /path/to/QP_data
export HPR_DATA_ROOT=/path/to/QP_data
```


## Tables, figures and scope

`./run.sh tables` regenerates tables without a GPU. Synthetic-CQP tables/profiles use the current selected HPR results, with archived competitor timings; other exports remain historical references. `Synthetic_HPR_R2_comparison.csv` provides the per-table numerical comparison. Shared-server verification timings are not an isolated cross-solver benchmark. For the 13 grayscale figure PDFs, install `environment/requirements-postprocess.txt` and run `./run.sh figures`. These table/figure exports remain in `experiments/paper_exports/`; the live experiment summaries above are the primary reviewer entry point.

The scope is the revised paper: 14 source benchmark groups, 365 case/curve checks, six source implementations, plus the two-method appendix example, grouped into ten descriptive experiments. [The paper result map](config/paper_result_map.csv) records the manuscript labels. The manuscript itself is submitted separately; the scripts use explicit reference tables in `config/reference/`.

## Package contents

| Folder | Purpose |
|---|---|
| `src/` | Required HPR-QP variants and competing-solver sources, licenses and patches |
| `environment/` | Pinned Julia environments and plotting dependencies |
| `scripts/` | Setup, execution, data acquisition, result reporting and checks |
| `config/` | Paper cases/settings, reference values and packaged verification index |
| `data/` | Bundled inputs, full data inventories, archive index and acquisition instructions |
| `experiments/` | Results organized by experiment and solver, including native evidence |
| `docs/` | Detailed experiment costs, environment, validation and third-party notices |

## Support

For questions about this reproduction package, contact the authors listed in [AUTHORS](AUTHORS). Third-party software notices are in [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md).
