# Reproduced results

Each `exp_<name>/solver_<name>/summary.csv` contains available rerun measurements, including unsuccessful outcomes. `raw/` holds the dataset CSV/log output or competitor-native records, and `statistics.csv` contains aggregates. New attempts keep their native output in `raw/runs/<attempt>/native/` without overwriting distributed raw evidence. HPR raw data uses the `run_dataset` column names; normalized summaries retain each measurement's source path.

Each experiment's `summary.csv` has two consecutive result sets, `paper` and `reproduced`, identified by `result_set`. Measured and pending case counts expose incomplete coverage; missing aggregate values say `to be reproduced`. Partial solved counts refer only to measured cases.

Each experiment's `comparison.csv` contains only observed differences, with `differences`, `expected` and `note` columns. It omits matching results, runtime-only differences and unrun cases. Timings remain in solver summaries and appear as context on retained comparison rows. A header-only comparison means no differences were flagged among the available reruns, not that every case has been reproduced. Original paper data is never substituted into a reproduced measurement. Timing aggregates and solver profiles are withheld for incomplete datasets or external execution failures.

The restart/sigma experiment uses `fig_Restart-Sigma/` and CSV curve samples. Corrected synthetic models remain labeled in the comparison; historical objectives are not silently changed. The appendix example agrees with the approximate iteration counts in Online Supplement Appendix E.

Recent focused compatibility/validation checks are excluded. See [interpretation](../../docs/VALIDATION.md) and the [result guide](../README.md).
