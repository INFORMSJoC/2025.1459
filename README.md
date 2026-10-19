[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# HPR-QP: A dual Halpern Peaceman–Rachford method for solving large scale convex composite quadratic programming

Kaihuang Chen, Defeng Sun, Yancheng Yuan, Guojun Zhang and Xinyuan Zhao.

This archive accompanies the [paper](https://doi.org/10.1287/ijoc.2025.1459) in the INFORMS Journal on Computing. It contains the HPR-QP reproduction snapshot, experiment scripts, and numerical records. The authors' software is distributed under the [MIT License](LICENSE); third-party software retains its own licenses. Dataset origins and terms are documented [by instance type](data/README.md).

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

## Description

HPR-QP implements a dual Halpern Peaceman–Rachford method for large-scale convex composite quadratic programming. The experiments compare the dual and primal variants, benchmark other solvers, and examine implicit formulations, trust-region problems, restart and penalty updates.

| Directory | Contents |
|---|---|
| [src](src/README.md) | HPR-QP variants and third-party solver sources/licenses |
| [data](data/README.md) | One directory per instance type, exact instance lists, origins, terms and acquisition instructions |
| [scripts](scripts/README.md) | Launcher, data tools, postprocessing, settings and pinned environments |
| [results/paper_results](results/paper_results/README.md) | Tables, figures, paper mapping and per-experiment summaries |
| [results/raw_results](results/raw_results/README.md) | Archived numerical records, native verification output and new-run records |
| [docs](docs/README.md) | Detailed experiment, environment, validation and attribution notes |

## Building

Use Linux, Bash, Python 3.11+, Julia **1.10.4**, and a compatible NVIDIA GPU for GPU solvers. Gurobi requires your own license. Dependency versions are pinned under [scripts/environments](scripts/environments/README.md); see [environment details](docs/ENVIRONMENT.md). Do not update the manifests for paper reproduction.

```bash
export JULIA_BIN=/path/to/julia-1.10.4/bin/julia
./scripts/run.sh check            # Package checks; no data download or GPU required
./scripts/run.sh setup HPR-QP     # Install pinned dependencies
```

## Results

Start with the [paper results guide](results/paper_results/README.md). The [paper-to-result map](results/paper_results/paper_result_map.csv) identifies the manuscript tables and figures. [Tables and figures](results/paper_results/tables_and_figures/) are separate from [raw records](results/raw_results/README.md).

For example, open the [QAPLIB summary](results/paper_results/experiments/QAPLIB/summary.html), [primal–dual summary](results/paper_results/experiments/Primal-Dual/summary.html), or [synthetic CQP summary](results/paper_results/experiments/Synthetic-CQP/summary.html). Each has a CSV equivalent. These summaries use available fresh measurements and explicitly label archived fallback; a mixed summary is **not** a full fresh benchmark. Every selected row identifies its numerical source.

The preserved verification evidence covers 365 HPR benchmark/curve checks, including time-limit outcomes. Competitors have sampled verification across applicable datasets. Existing numerical limitations and manuscript differences remain visible in [validation](docs/VALIDATION.md).

## Replicating

The launcher is now **`./scripts/run.sh EXPERIMENT SOLVER`**. Run commands from the repository root, or invoke the launcher by its absolute path. Command-line data/output paths remain relative to your working directory.

Third-party MM and QAPLIB input files are no longer bundled. First read the [data acquisition guide](data/README.md). Existing prepared archives remain available through the previously supplied Dropbox link **while the editor considers whether this hosting is acceptable**; this is not a claim of approved permanent hosting or a grant of third-party rights. Original sources and exact instance lists are provided in each dataset directory.

For the compact, complete 36-case QAPLIB experiment, obtain the prepared MAT archive as described in [data/QAPLIB](data/QAPLIB/README.md), then:

```bash
python3 scripts/manage_data.py extract --groups QAPLIB --cache /path/to/2025.1459 --data-root /path/to/data
./scripts/run.sh QAPLIB HPR-QP --data-root /path/to/data --plan
./scripts/run.sh QAPLIB HPR-QP --data-root /path/to/data
```

The complete QAPLIB HPR-QP run historically takes about 40–90 minutes on the documented hardware. The [experiment guide](docs/EXPERIMENTS.md) gives additional costs and choices. A fully bundled small example is available with `./scripts/run.sh Appendix-Example HPR-QP` after setup; it is not a replacement for the benchmark experiments.

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

`HPR-QP` uses the implicit implementation for Lasso and QAP. Standard explicit CQPs use `src/variants/main`; `HPR-QP-explicit` uses that main implementation for explicit Lasso, preserving the disabled diagonal-Q shortcut. `HPR-p1` and `HPR-p2` are the primal variants. `src/variants` contains the authors' implementations developed specifically for this project and used only in this project; see [provenance](src/variants/README.md).

```bash
./scripts/run.sh list
./scripts/run.sh MM HPR-QP --cases CVXQP3_S.mps QSCSD1.mps --data-root /path/to/data
./scripts/run.sh ALL HPR-QP --data-root /path/to/data --plan
./scripts/run.sh refresh   # Refresh summaries from available records; no solves
./scripts/run.sh tables    # Rebuild CSV tables; no GPU
./scripts/run.sh figures   # Rebuild grayscale PDFs; plotting dependencies required
```

Outputs are written under `results/raw_results` and `results/paper_results`; `--output /path/to/results` selects an alternative results root. New runs retain native logs/settings and update the corresponding summary. Full `ALL ALL` runs can take days to weeks. Use `--help` for time, memory and device controls. For figures, install `scripts/environments/requirements-postprocess.txt`.

## Ongoing Development

The ongoing development repository is [PolyU-IOR/HPR-QP](https://github.com/PolyU-IOR/HPR-QP). This journal repository preserves the paper reproduction snapshot.

## Support

For questions about this submission, open an [issue](https://github.com/INFORMSJoC/2025.1459/issues) or contact the authors listed in [AUTHORS](AUTHORS).
