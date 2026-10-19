# Paper tables, figures and experiment summaries

[paper_result_map.csv](paper_result_map.csv) maps the manuscript tables and figures to experiments.

- [tables_and_figures](tables_and_figures/): the CSV exports and `figures/` PDFs previously called `paper_exports`.
- `experiments/<experiment>/summary.html` and `summary.csv`: solved counts and SGM10, with explicit fresh/archived source counts.
- `experiments/<experiment>/results.csv`: selected per-case measurements and raw-source paths.
- `experiments/<experiment>/comparison.csv`: fresh-versus-archived differences.
- `experiments/<experiment>/<solver>/{archived,fresh}.csv`: solver-specific tables.

Available experiments: [Primal-Dual](experiments/Primal-Dual/summary.html), [MM](experiments/MM/summary.html), [Synthetic-CQP](experiments/Synthetic-CQP/summary.html), [Lasso-UCI](experiments/Lasso-UCI/summary.html), [Lasso-Synthetic](experiments/Lasso-Synthetic/summary.html), [QAPLIB](experiments/QAPLIB/summary.html), [Large-QAP](experiments/Large-QAP/summary.html), [Trust-Region](experiments/Trust-Region/summary.html), [Restart-Sigma](experiments/Restart-Sigma/summary.html), [Appendix-Example](experiments/Appendix-Example/summary.html).

Every available fresh row replaces its archived row, including unsuccessful solves; unrun rows use archived values. Mixed summaries are not complete fresh benchmarks. Execution failures withhold timing aggregates. Original numerical discrepancies, corrected-model checks and source labels remain visible.

Synthetic-CQP exports use selected corrected-model HPR measurements and archived competitor timings. Other exports remain historical references. `Synthetic_HPR_R2_comparison.csv` reports differences from the manuscript; the manuscript itself has not been changed by this layout revision.

Regenerate from the repository root with `./scripts/run.sh refresh`, `tables`, or `figures`.
