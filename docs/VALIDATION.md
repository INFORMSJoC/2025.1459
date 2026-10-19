# Result coverage and interpretation

The package covers the revised paper's ten named experiments. The [README experiment table](../README.md#replicating) maps each command to its manuscript table or figure.

The preserved HPR measurements cover 365 benchmark and curve checks, including time-limit outcomes. The two-method appendix example is also included. Competing solvers have selected rerun measurements across applicable dataset combinations; these are not complete reruns of every dataset. Completed coverage does not mean that every case converged or every value equals its paper reference.

## Reading the results

Use [the result guide](../results/README.md). Original paper values and reruns are stored separately by experiment and solver. Each reproduced experiment's `summary.csv` presents the `paper` result set followed by `reproduced`. Incomplete aggregate cells say `to be reproduced`, with measured and pending counts beside them. Solver `summary.csv` files hold actual per-case measurements and `statistics.csv` files hold aggregates. Raw records reside with their solver.

Each `comparison.csv` contains only observed differences. Runtime-only changes and unrun cases are excluded; their absence is not proof of reproduction. Iteration counts are compared exactly when both are recorded. Solved-status synonyms are treated as equivalent, and `REFERENCE_CURVE` is a reference label rather than a solver termination. Objectives on the same recorded basis are flagged when `abs(reproduced - paper) / max(1, abs(paper)) > 1e-6`, using the existing validation threshold. This screens differences; it is not an optimality guarantee. Missing or incomparable objective values are not treated as disagreements. Curve mismatches are included when the runner's saved curve check reports one.

The `expected` column is `yes` only when a documented mechanism explains the change: corrected model, different time-limit endpoint. `consistent with rounding` means preserved CuClarabel objectives agree to five significant digits; this is a precision comparison. `not established` means at least one difference lacks a confirmed explanation. Matching objectives alone do not prove why iteration counts changed. Every retained row includes its explanation and source paths.

SGM10 is `exp(mean(log(x + 10))) - 10`. Ordinary failures use the 3,600-second cap; Large-QAP uses its configured 360,000-second cap. Missing tolerance checkpoints remain blank. An incomplete dataset, external time/memory guard, interruption or execution failure withholds the full timing aggregate. Partial competitor data is excluded from full-dataset performance profiles.

Solver residual definitions differ, so objectives and termination information should be read together with native records. Some competitor references contain rounded objectives; a small difference from the unrounded rerun need not indicate disagreement at paper precision. Iteration equality is reported separately from objective agreement. Curve comparisons use the saved iteration grid and pointwise log10 residual deviation at most 0.01; a no-restart curve can match without reaching the stopping tolerance.

## Known differences

At tolerance 1e-8, the 128 converged MM HPR cases agree with their reference objectives; the other nine reach the time limit in both records. HPR objective comparisons also agree for the 36 QAPLIB, six Large-QAP, 11 Lasso-UCI, 12 Lasso-Synthetic and 17 Trust-Region cases. Timing and occasional iteration differences remain visible.

The denser synthetic HPR runs use the corrected original-MPS objective. Their historical old-loader objectives are not valid targets for that formulation. Paper values remain unchanged in `paper_results`; corrected reruns remain in `reproduced_results`.

The appendix example gives 900 dual and 1,700 primal iterations, consistent with R2 Online Supplement Appendix E. Objectives and the four matrix norms agree at the paper's precision.

Preserved CuClarabel benchmark measurements retain their original environment information, including earlier compatibility-patched runs. The executable package now uses unmodified CuClarabel with its compatible Julia 1.11.5 environment. These old timings are not relabeled as measurements of the current environment. Separate development validation checks are excluded from the distributed results.

## Running and checking

`./scripts/run.sh check` verifies release hashes, experiment/data scope, native-source links, result layout and relative Julia dependencies. `scripts/manage_data.py verify` checks external benchmark files. The inventory describes 315 input representations.

`./scripts/run.sh refresh`, `tables` and `figures` rebuild reports from the separate paper and rerun CSVs. The launcher uses the dataset writers for HPR benchmark output and CSV for launch status/normalized measurements. The figure driver writes curve CSVs. Competitor-native records retain their original formats. See [environment details](ENVIRONMENT.md) for dependencies and [experiment costs](EXPERIMENTS.md) for planning.
