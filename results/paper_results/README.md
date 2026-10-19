# Original paper results

Each `exp_<name>/solver_<name>/summary.csv` contains the original paper's per-case measurements. `raw/` holds the underlying solver tables; shared source tables are in the experiment's `sources/`. Some historical logs were not preserved and are not fabricated. Solver `statistics.csv` reports calculated aggregates. Experiment-level `summary.csv` uses the printed R2 aggregates from `sources/reported_statistics.csv` where available.

The figure experiment uses `fig_Restart-Sigma/`. Its original curve data and drawing are preserved. Other experiment figures are regenerated from paper measurements; corrections to known joins/statuses retain the original source tables for inspection.

Reruns and paper-versus-rerun comparisons belong to [reproduced_results](../reproduced_results/README.md). See the [result guide](../README.md) for the full layout.
