# Experimental results

- [paper_results](paper_results/README.md): original paper measurements.
- [reproduced_results](reproduced_results/README.md): rerun measurements and comparisons.
- [paper_result_map.csv](paper_result_map.csv): manuscript table/figure mapping.

Both result trees use the same layout:

```text
exp_MM/
  summary.csv
  solver_HPR-QP/
    summary.csv
    statistics.csv
    raw/
      HPRQP_zxwyw_result.csv
      HPRQP_log.txt
  figures/
```

`solver_*/summary.csv` contains per-case measurements, and `statistics.csv` reports aggregate statistics. In `paper_results`, the experiment-level `summary.csv` uses printed R2 aggregates where available, stored in `sources/reported_statistics.csv`. Solver `statistics.csv` files retain aggregates calculated from the per-case measurements. In `reproduced_results`, it contains two consecutive result sets distinguished by `result_set`: `paper` followed by `reproduced`. Pending reruns are counted explicitly, and incomplete aggregate cells say `to be reproduced`. Solved counts on partial rows cover measured cases only. A metric absent from the reference or not defined for an experiment says `not available`; execution failures remain visible and withhold full aggregates.

`reproduced_results/exp_*/comparison.csv` lists only observed differences in outcomes, termination, iterations, objectives, or checked curves. Each row states whether its differences are expected and why. Runtime-only changes, matching results and unrun cases are omitted. An empty comparison has a header only; it does not imply full rerun coverage. Neither solver summaries nor plots substitute paper values for missing reruns. See [comparison criteria](../docs/VALIDATION.md#reading-the-results).

The restart/sigma experiment uses `fig_Restart-Sigma/`. Its solver's `raw/curves.csv` contains iteration/residual samples. Multi-solver plots and original shared source tables belong to their experiment. Groups sharing instance names, such as standard and denser synthetic cases, have separate raw subdirectories.

HPR benchmark runs use the original dataset CSV/log writers. Existing preserved HPR measurements have been transferred losslessly from the former verification wrapper into the same tabular schema. HPR result folders contain no JSON. Competitor-native JSON is retained where that is the solver's recorded format. Environment/settings information is CSV or solver log text.

Run `./scripts/run.sh refresh`, `tables`, or `figures` to regenerate reports. Local attempts keep their native CSV/logs under the selected solver's `raw/runs/<attempt>/native/`; summaries select the latest attempt and link its native record. Distributed raw evidence remains unchanged so release integrity checks continue to work after reruns; temporary job configuration is outside the result trees and excluded from releases. Validation-only checks are not part of the package.
