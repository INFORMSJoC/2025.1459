# Original paper results

Each `exp_<name>/solver_<name>/summary.csv` contains the original paper's per-case measurements. `raw/` holds the underlying solver tables; shared source tables are in the experiment's `sources/`. Some historical logs were not preserved and are not fabricated. Solver `statistics.csv` reports calculated aggregates. Experiment-level `summary.csv` uses the printed paper aggregates from `sources/reported_statistics.csv` where available.

The figure experiment uses `fig_Restart-Sigma/`. Its original curve data are preserved; `QSCSD1.pdf` uses the colored figure from [the reproduced curves](../reproduced_results/fig_Restart-Sigma/QSCSD1.pdf). Other experiment figures are regenerated from paper measurements; corrections to known joins/statuses retain the original source tables for inspection.

Reruns and paper-versus-rerun comparisons belong to [reproduced_results](../reproduced_results/README.md). See the [result guide](../README.md) for the full layout.

## Figure locations

The main paper has one figure. Online Supplement Figures S1–S4 each contain three tolerance panels, stored as separate PDFs below. Experiments that report tables only do not need a `figures/` directory.

| Paper figure | Experiment | PDF files |
|---|---|---|
| Figure 1 | Restart and penalty updates | [QSCSD1](fig_Restart-Sigma/QSCSD1.pdf) |
| Online Supplement Figure S1 | MM performance profiles | [1e-4](exp_MM/figures/profile_1e-04.pdf), [1e-6](exp_MM/figures/profile_1e-06.pdf), [1e-8](exp_MM/figures/profile_1e-08.pdf) |
| Online Supplement Figure S2 | Synthetic CQP performance profiles | [1e-4](exp_Synthetic-CQP/figures/profile_1e-04.pdf), [1e-6](exp_Synthetic-CQP/figures/profile_1e-06.pdf), [1e-8](exp_Synthetic-CQP/figures/profile_1e-08.pdf) |
| Online Supplement Figure S3 | QAPLIB performance profiles | [1e-4](exp_QAPLIB/figures/profile_1e-04.pdf), [1e-6](exp_QAPLIB/figures/profile_1e-06.pdf), [1e-8](exp_QAPLIB/figures/profile_1e-08.pdf) |
| Online Supplement Figure S4 | Trust-region solve times | [1e-4](exp_Trust-Region/figures/solve_times_1e-04.pdf), [1e-6](exp_Trust-Region/figures/solve_times_1e-06.pdf), [1e-8](exp_Trust-Region/figures/solve_times_1e-08.pdf) |

Run `./scripts/run.sh figures` to regenerate the performance and scaling plots from the saved measurements. The restart experiment retains this saved presentation copy and regenerates the reproduced figure from its curve samples.
