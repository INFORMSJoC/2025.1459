# Experiments, recommendations and cost

All benchmark inputs are external; follow [data acquisition](../data/README.md) first. Only the appendix example and initialization support input are bundled. Existing `QP_data` paths remain supported.

Run a **complete named experiment for one solver** with `./scripts/run.sh NAME SOLVER`. Recommendations below are for whole datasets/experiments, not a selection of easy instances. A recommendation applies only to the solver named in that row. Every run uses all paper tolerances by default: HPR-QP and PDQP record crossings in one solve; the JuMP solvers run separately at each tolerance.

Cost estimates are listed in [experiment_costs.csv](experiment_costs.csv), covering every supported experiment/solver combination. Solver minutes are **sums, not SGM10**. Loading, model construction, warmup and compilation can dominate. The estimates below assume sequential execution on the tested A100 80 GB server with installed dependencies; downloads/installations are extra. They are planning ranges, not measured end-to-end runs. Measured small-worker wall times were approximately 118 seconds for HPR-QP/MM and 60 seconds for HPR-QP/QAPLIB. Each new worker repeats startup and warmup.

| Command name | Revised-paper experiment / location | Cases per solver | HPR-QP input size | HPR-QP cost evidence and estimated elapsed time | Recommendation / complete command |
|---|---|---:|---:|---|---|
| `QAPLIB` | R2 Table 6; Online Supplement Appendix D.2 (performance profiles) | 36 | 4 MB; download QAPLIB MAT archive | 4.1 min archived solve total; 4.5 min fresh loading + solve sections. **Allow 40–90 min** with separate worker startup/warmup. | **Recommended complete dataset:** `./scripts/run.sh QAPLIB HPR-QP` |
| `Primal-Dual` | R2 Table 1; Online Supplement Table S1 | 30 for each variant | 60 MB; download MM archive | HPR-QP: 4.3 min solve total; **allow 1–2 h**. HPR-p1/p2: 23.1/26.3 min solve totals, plus 30 startups each. | **Recommended HPR-QP experiment:** `./scripts/run.sh Primal-Dual HPR-QP --data-root /path/to/QP_data`. Use `ALL` for all three variants; allow several hours. |
| `Restart-Sigma` | R2 Figure 1 | 3 curves | QSCSD1; download MM archive | Fresh measured curve sections total about 5 s, excluding startup/warmup. **Allow 5–15 min.** | **Recommended complete figure experiment:** `./scripts/run.sh Restart-Sigma HPR-QP` |
| `Lasso-UCI` | R2 Table 4; Online Supplement Tables S3 (dimensions) and S4 (results) | 11 | 15.8 GB; external | Implicit HPR-QP: 8.5 min archived solves; 17.7 min fresh loading + solve sections. **Allow 30–90 min**, with additional data and sufficient memory. Explicit HPR: 82 min fresh sections before startup/warmup. | **Optional complete dataset if the data/memory are available:** `./scripts/run.sh Lasso-UCI HPR-QP --data-root /path/to/QP_data` |
| `MM` | R2 Table 2; Online Supplement Appendix D.2 (performance profiles) | 137 | 194 MB; download MM archive | 10.0 h archived solver total, including 9 timeout cases; fresh sections 10.1 h. Startup adds several hours. | Full benchmark, **not a modest-cost recommendation**: `./scripts/run.sh MM HPR-QP --data-root /path/to/QP_data` |
| `Synthetic-CQP` | R2 Table 3; Online Supplement Table S2 and Appendix D.2 (performance profiles) | 60 | 198.4 GB; external | 78.8 min archived solves, but fresh loading + solve sections total **4.8 h** before warmup/startup. Large memory/storage cost. | Full benchmark: `./scripts/run.sh Synthetic-CQP HPR-QP --data-root /path/to/QP_data` |
| `Lasso-Synthetic` | R2 Table 5; Online Supplement Table S5 | 12 | 228.0 GB; external | Only 7.4 min archived implicit solves, but **3.6 h** fresh loading + solve sections; explicit sections **5.2 h**. Warmup/startup are extra. | Full benchmark, not recommended for a short review: `./scripts/run.sh Lasso-Synthetic HPR-QP --data-root /path/to/QP_data` |
| `Trust-Region` | R2 Table 8; Online Supplement Figure S4 | 17 | 31.5 GB; external | 5.4 min archived solves; **75.5 min** fresh loading + solve sections, before repeated loading in warmup. Large matrices. | Full benchmark: `./scripts/run.sh Trust-Region HPR-QP --data-root /path/to/QP_data` |
| `Large-QAP` | R2 Table 7 | 6 | 2.9 GB; external | **38.0 h** archived solver total; largest case alone about 35 h. GPU memory grows far beyond the MAT file size. | Expensive full benchmark: `./scripts/run.sh Large-QAP HPR-QP --data-root /path/to/QP_data` |
| `Appendix-Example` | Online Supplement Appendix E (as referenced by R2) | 1 per method | Tiny bundled JSON | Allow a few minutes for startup/compilation. | Run both methods: `./scripts/run.sh Appendix-Example ALL`. |

Input size is uncompressed disk size, **not peak RAM/GPU memory**. All costs refer to the complete indicated set. The smallest cases are not used to label an otherwise expensive dataset inexpensive. Recorded loading/solve sections exclude Julia startup and separate warmup; allow additional time for these in each worker.

## Solver choices and full competing-solver cost

| Experiment | Available solver arguments |
|---|---|
| Primal-Dual | `HPR-QP`, `HPR-p1`, `HPR-p2` |
| MM, Synthetic-CQP, QAPLIB | `HPR-QP`, `PDQP`, `SCS`, `CuClarabel`, `Gurobi` |
| Lasso-UCI, Lasso-Synthetic | Above five, plus `HPR-QP-explicit` |
| Trust-Region | `HPR-QP`, `SCS`, `CuClarabel`, `Gurobi` |
| Large-QAP, Restart-Sigma | `HPR-QP` |
| Appendix-Example | `HPR-QP`, `HPR-p1` |

`ALL` as the solver means all solvers **reported for that experiment**. `ALL` as the experiment means every applicable experiment for the selected solver. In `ALL ALL`, shared HPR-QP MM/Primal-Dual cases are computed once and used in both reports.

The following totals are archived solver-time sums for the default tolerance protocol, in **hours**, excluding setup/loading/warmup. Native failures are charged the paper's 3,600-second cap for planning; memory failures do not necessarily consume that duration. Real elapsed times can differ substantially.

| Complete dataset | PDQP | SCS | CuClarabel | Gurobi |
|---|---:|---:|---:|---:|
| MM | 17.1 | 107.1 | 21.0 | 2.1 |
| Synthetic-CQP | 6.1 | 113.0 | 40.5 | 80.4 |
| Lasso-UCI | 2.5 | 27.3 | 0.6 | 1.6 |
| Lasso-Synthetic | 1.5 | 36.0 | 30.0 | 30.1 |
| QAPLIB | 13.4 | 3.7 | 17.2 | 3.9 |
| Trust-Region | Not reported | 19.7 | 33.5 | 45.6 |

No complete competitor dataset is labeled a low-cost verified recommendation: only selected-case competitor reruns have been performed. CuClarabel on all UCI data is the most plausible modest full competitor experiment, but loading and memory cost must still be allowed for. QAPLIB competitors require **53.4 GB of explicit MPS inputs**, rather than HPR-QP's compact MAT data. No explicit QAP MPS inputs are bundled; all 36 checksummed locations are in `scripts/config/data/competitor_qap_manifest.json`.

## Optional instance-wise checks

The normal launcher also supports exact case names; this is independent of the complete-dataset recommendations above:

```bash
./scripts/run.sh MM HPR-QP --cases CVXQP3_S.mps QSCSD1.mps
./scripts/run.sh MM CuClarabel --cases CVXQP3_S.mps QSCSD1.mps
./scripts/run.sh QAPLIB SCS --cases esc64a.mps --tolerances 1e-8 --data-root /path/to/QP_data
./scripts/run.sh MM HPR-QP --cases CVXQP3_S.mps QSCSD1.mps --plan
```

Synthetic case names can repeat in the original and denser datasets; use `--groups synthetic_paper` or `--groups revision_denser` to distinguish them. Exact canonical keys are also listed by `--plan`. `--tolerances` selects reported checkpoints for HPR-QP/PDQP; these solvers still run to the original final stopping tolerance. For JuMP solvers it selects independent solves.

## Long runs, interruption and devices

```bash
./scripts/run.sh ALL HPR-QP --data-root /path/to/QP_data --plan
./scripts/run.sh ALL ALL --data-root /path/to/QP_data
# Optional parallel solver workers across unused GPUs:
./scripts/run.sh MM ALL --gpus 1,2,3 --jobs 3
```

**Full ALL can take days to weeks.** With all external data installed, the complete plan selects 3,080 Julia workers across ten experiments. Sequential archive time-cap accounting sums to about 706 hours (29 days), before startup/warmup; early memory failures can finish sooner, so this is a planning total rather than a measured elapsed-time prediction. Unsuccessful competitor cases can run to their deadline or exhaust available memory. Every Julia worker has an external wall-clock guard, including loading, compilation and warmup. Default competitor guard: native limit + 900 seconds (normally 4,500 seconds). HPR's default guard allows native limit + 14,400 seconds because large MPS parsing/warmup can take hours; Large-QAP retains its 360,000-second native limit. Use `--hard-time-limit SECONDS` for an explicit per-worker budget, `--run-time-limit SECONDS` for the invocation, and optionally `--memory-gb N` for a sampled host-RSS guard. The RSS guard is not a GPU-memory cap or a guarantee against an OOM spike.

Ctrl-C stops active process groups (TERM, then KILL) and retains finished work and interruption records. Resume with the same output directory; completed matching runs are skipped. Failed attempts are retried. `--rerun` repeats completed attempts. Runtime-limit, memory-limit and execution failures stay visible and withhold the effective timing aggregate; they are not silently replaced by archived values.

Default execution is sequential. Within one package checkout, each selected GPU is locked to one worker and only one Gurobi worker runs at a time. Reserve GPUs not used by unrelated processes or other package copies. GPU solvers also use CPU preprocessing; multiple CPU-intensive tasks compete for cores and memory bandwidth. Use sequential execution on an otherwise idle machine for timing comparisons. Gurobi defaults to 32 threads; changing `--cpu-threads` is recorded and changes the performance environment.
