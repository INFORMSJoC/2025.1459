[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# HPR-QP: A dual Halpern Peaceman–Rachford method for solving large scale convex composite quadratic programming

Kaihuang Chen, Defeng Sun, Yancheng Yuan, Guojun Zhang and Xinyuan Zhao.

This archive accompanies the [paper](https://doi.org/10.1287/ijoc.2025.1459) in the INFORMS Journal on Computing. It contains the HPR-QP reproduction snapshot, experiment scripts, and numerical records. The authors' software is distributed under the [MIT License](LICENSE); third-party software retains its own licenses. Dataset origins and terms are documented [by instance type](data/README.md).

## Cite

Please cite both the paper and this repository using their respective DOIs:

- Paper: [10.1287/ijoc.2025.1459](https://doi.org/10.1287/ijoc.2025.1459)
- Repository: [10.1287/ijoc.2025.1459.cd](https://doi.org/10.1287/ijoc.2025.1459.cd)

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

## Description

HPR-QP implements a dual Halpern Peaceman–Rachford method for large-scale convex composite quadratic programming. The experiments compare the dual and primal variants, benchmark other solvers, and examine implicit formulations, trust-region problems, restart and penalty updates.

| Directory | Contents |
|---|---|
| [src](src/README.md) | HPR-QP variants and third-party solver sources/licenses |
| [data](data/README.md) | One directory per instance type, exact instance lists, origins, terms and acquisition instructions |
| [scripts](scripts/README.md) | Launcher, data tools, postprocessing, settings and pinned environments |
| [results/paper_results](results/paper_results/README.md) | Original paper measurements, organized by experiment and solver |
| [results/reproduced_results](results/reproduced_results/README.md) | Rerun measurements, raw CSV/log output and comparisons with the paper |
| [docs](docs/README.md) | Detailed experiment, environment, validation and attribution notes |

## Building

Use Linux, Bash, Python 3.11+, Julia **1.10.4** (Julia **1.11.5** for CuClarabel), and a compatible NVIDIA GPU for GPU solvers. Gurobi requires your own license. Dependency versions are pinned under [scripts/environments](scripts/environments/README.md); see [environment details](docs/ENVIRONMENT.md). Do not update the manifests for paper reproduction.

```bash
export JULIA_BIN=/path/to/julia-1.10.4/bin/julia
export CUCLARABEL_JULIA_BIN=/path/to/julia-1.11.5/bin/julia
# If python3 is older than 3.11, select an installed newer interpreter:
# export PYTHON_BIN=/path/to/python3.11
./scripts/run.sh check            # Package checks; no data download or GPU required
./scripts/run.sh setup HPR-QP     # Install pinned dependencies
```

CuClarabel uses its original source with a separate CUDA 12.8 environment, tested on NVIDIA driver 570.133.20. See [environment details](docs/ENVIRONMENT.md) before running it on a newer driver.

## Results

The [result guide](results/README.md) describes the parallel `paper_results/` and `reproduced_results/` trees. The [paper-to-result map](results/paper_result_map.csv) identifies the manuscript tables and figures.

For example, the [QAPLIB comparison](results/reproduced_results/exp_QAPLIB/comparison.csv) lists observed differences and explains whether they are expected. Runtime-only differences and unrun cases are omitted. Within each experiment, `solver_<name>/summary.csv` contains per-case values, `statistics.csv` contains solved counts and SGM10, and `raw/` holds the underlying output. The reproduced experiment's `summary.csv` has two labeled result sets, `paper` then `reproduced`; incomplete rerun aggregates are marked `to be reproduced`, with measured and pending case counts. Paper values never fill missing rerun measurements.

The preserved verification evidence covers 365 HPR benchmark/curve checks, including time-limit outcomes. Competitors have sampled verification across applicable datasets. Existing numerical limitations and manuscript differences remain visible in [validation](docs/VALIDATION.md).

## Replicating

The launcher is **`./scripts/run.sh EXPERIMENT SOLVER`**. Run commands from the repository root, or invoke the launcher by its absolute path. Command-line data/output paths remain relative to your working directory.

Prepared experiment inputs are hosted on Dropbox. See the [data acquisition guide](data/README.md) for download and extraction instructions, dataset sources, and applicable terms.

For the compact, complete 36-case QAPLIB experiment, obtain the prepared MAT archive as described in [data/QAPLIB](data/QAPLIB/README.md), then:

```bash
python3 scripts/manage_data.py extract --groups QAPLIB --cache /path/to/2025.1459 --data-root /path/to/data
./scripts/run.sh QAPLIB HPR-QP --data-root /path/to/data --plan
./scripts/run.sh QAPLIB HPR-QP --data-root /path/to/data
```

The complete QAPLIB HPR-QP run historically takes about 40–90 minutes on the documented hardware. The [experiment guide](docs/EXPERIMENTS.md) gives additional costs and choices. A fully bundled small example is available with `./scripts/run.sh Appendix-Example HPR-QP` after setup; it is not a replacement for the benchmark experiments.

| `EXP_NAME` | Paper experiment / reference | Cases per solver | Supported `SOLVER` arguments |
|---|---|---:|---|
| `Primal-Dual` | Table 1; Online Supplement Table S1 | 30 | `HPR-QP`, `HPR-p1`, `HPR-p2` |
| `MM` | Table 2; Online Supplement Figure S1 | 137 | `HPR-QP`, `PDQP`, `SCS`, `CuClarabel`, `Gurobi` |
| `Synthetic-CQP` | Table 3; Online Supplement Table S2 and Figure S2 | 60 | `HPR-QP`, `PDQP`, `SCS`, `CuClarabel`, `Gurobi` |
| `Lasso-UCI` | Table 4; Online Supplement Tables S3 (dimensions) and S4 (results) | 11 | `HPR-QP`, `HPR-QP-explicit`, `PDQP`, `SCS`, `CuClarabel`, `Gurobi` |
| `Lasso-Synthetic` | Table 5; Online Supplement Table S5 | 12 | `HPR-QP`, `HPR-QP-explicit`, `PDQP`, `SCS`, `CuClarabel`, `Gurobi` |
| `QAPLIB` | Table 6; Online Supplement Figure S3 | 36 | `HPR-QP`, `PDQP`, `SCS`, `CuClarabel`, `Gurobi` |
| `Large-QAP` | Table 7 | 6 | `HPR-QP` |
| `Trust-Region` | Table 8; Online Supplement Figure S4 | 17 | `HPR-QP`, `SCS`, `CuClarabel`, `Gurobi` |
| `Restart-Sigma` | Figure 1 | 3 curves | `HPR-QP` |
| `Appendix-Example` | Online Supplement Appendix E | 1 | `HPR-QP`, `HPR-p1` |

`HPR-QP` uses the implicit implementation for Lasso and QAP. Standard explicit CQPs use `src/variants/main`; `HPR-QP-explicit` uses that main implementation for explicit Lasso, preserving the disabled diagonal-Q shortcut. `HPR-p1` and `HPR-p2` are the primal variants. `src/variants` contains the authors' implementations developed specifically for this project and used only in this project; see [provenance](src/variants/README.md).

```bash
./scripts/run.sh list
./scripts/run.sh MM HPR-QP --cases CVXQP3_S.mps QSCSD1.mps --data-root /path/to/data
./scripts/run.sh ALL HPR-QP --data-root /path/to/data --plan
./scripts/run.sh refresh   # Refresh summaries from available records; no solves
./scripts/run.sh tables    # Rebuild CSV tables; no GPU
./scripts/run.sh figures   # Rebuild color performance/scaling PDFs; plotting dependencies required
```

Outputs are written under `results/reproduced_results` and `results/paper_results`; `--output /path/to/results` selects an alternative results root. New runs retain native logs/settings and update the corresponding summary. Full `ALL ALL` runs can take days to weeks. Use `--help` for time, memory and device controls. For figures, install `scripts/environments/requirements-postprocess.txt` into your Python environment and select that interpreter with `PYTHON_BIN` when running `./scripts/run.sh figures`.

## Ongoing Development

The ongoing development repository is [PolyU-IOR/HPR-QP](https://github.com/PolyU-IOR/HPR-QP). This journal repository preserves the paper reproduction snapshot.

## Support

For questions about this package, open an [issue](https://github.com/INFORMSJoC/2025.1459/issues) or contact the authors listed in [AUTHORS](AUTHORS).
