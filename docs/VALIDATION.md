# Reproduction coverage and result interpretation

The package covers the revised paper's ten named experiments. The [README experiment table](../README.md#experiment-names-and-solvers) maps each command to its manuscript table or figure.

Completed HPR measurements cover **365/365 benchmark and curve checks**, including time-limit outcomes. The two-method appendix example is also included. Competing solvers were tested on selected cases across all 26 applicable solver/dataset combinations; they have not all been rerun on complete datasets. Completed coverage does not mean that every case converged or that every value is identical to its paper reference.

## Reading the results

Start with `experiments/<name>/summary.html` or `summary.csv`. These report solved counts and SGM10. `results.csv` contains per-instance values and `comparison.csv` compares fresh and archived measurements. Each solver's `archived.csv` holds the paper reference values, while `fresh.csv` holds available rerun values. Native solver records and logs are retained under that solver's `raw/` directory. New runs write to `runs/<attempt>/` and update the experiment summary automatically.

Fresh results replace the corresponding archived rows, including unsuccessful outcomes. Unrun rows retain archived values, and mixed summaries identify their sources. External time or memory guard stops, interruptions and execution failures withhold the effective timing aggregate. Setup and input failures appear as attempt errors.

SGM10 is `exp(mean(log(x + 10))) - 10`. Standard experiment failures use the 3,600-second cap; Large-QAP uses its configured 360,000-second cap. Missing tolerance-checkpoint values remain blank. Solver residual definitions differ, so objectives and termination information should be read together with each solver's native records.

For ordinary converged HPR comparisons, the checks require OPTIMAL termination, a final residual at most 1.01e-8 and scaled objective difference at most 1e-6. Iteration equality is reported separately. Curve checks compare the saved iteration grid and pointwise log10 residuals, allowing a deviation of 0.01. A no-restart curve can match its reference without reaching the stopping tolerance. Runtime and occasional iteration differences can occur across executions; timeout endpoints also depend on machine throughput.

## Representative launcher results

These cases completed through `run.sh` on the tested A100 server. The ten workers took approximately 9.4 minutes in total, including startup and warmup. They used a shared machine, with one CPU thread for Gurobi; these sample times are workflow checks, not isolated performance measurements.

| Experiment | Solver | Case | Result |
|---|---|---|---|
| MM | HPR-QP (main) | CVXQP3_M.mps | 40,900 iterations; archived objective/count reproduced |
| QAPLIB | HPR-QP (implicit) | sko64.mat | 23,300 iterations; archived objective/count reproduced |
| Lasso-UCI | HPR-QP (implicit) | mpg_scale_expanded7.mps | 6,600 iterations; archived objective/count reproduced |
| Lasso-UCI | HPR-QP-explicit (main) | mpg_scale_expanded7.mps | 13,800 iterations; archived objective/count reproduced |
| Trust-Region | HPR-QP | d10000.mat | 500 iterations; archived objective/count reproduced |
| MM | PDQP | CVXQP3_S.mps | OPTIMAL, 7,488 iterations; objective/count agree |
| MM | SCS | CVXQP3_S.mps | OPTIMAL, 325 versus archived 300 iterations; objective agrees |
| MM | CuClarabel | CVXQP3_S.mps | OPTIMAL, 11 iterations; objective/count agree |
| MM | Gurobi | CVXQP3_S.mps | OPTIMAL, 13 iterations; objective agrees; reference CSV has no iteration count |
| QAPLIB | CuClarabel | esc64a.mps | OPTIMAL, 6 iterations; objective/count agree |

The numerical records, solver settings and environment information accompany the experiment results. See [environment details](ENVIRONMENT.md) for pinned versions and [experiment costs](EXPERIMENTS.md) for full-dataset planning estimates.

## Package checks

`./run.sh check` verifies release hashes, the 40 bundled input hashes, experiment scope, data-catalog consistency and relative Julia source dependencies. The manifests describe 315 inputs across HPR and explicit-QAP competitor formats. Full datasets are downloaded separately following the [data guide](../data/README.md).

The launcher has been checked for interruption, process cleanup, resuming completed runs, guard-failure reporting, paths containing spaces and split-archive extraction. Table and grayscale-figure generation is available through `./run.sh tables` and `./run.sh figures`. Solver records and per-instance comparisons remain available for inspecting numerical results.
