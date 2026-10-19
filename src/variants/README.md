# HPR-QP implementations and provenance

The authors confirm that the code in this directory was developed specifically for the HPR-QP project and has been used only in this project. It is covered by the authors' [MIT license](../../LICENSE). Julia dependencies and the separate competing-solver snapshots retain their own licenses.

| Directory | Role in the paper |
|---|---|
| `main` | Main dual HPR-QP implementation for explicit CQPs and the explicit Lasso comparison |
| `implicit` | Implicit Lasso and QAP formulations |
| `primal1` | First primal HPR variant |
| `primal2` | Second primal HPR variant |
| `tr_mat` | Second-order-cone trust-region implementation reading the archived MAT inputs |
| `figure_QSCSD1` | Implementation used for the restart/penalty comparison curves |

These are experiment-specific snapshots. The [historical source record](../../results/raw_results/reference/source_provenance.json) identifies collected source files and portability patches. Its hashes describe the collected snapshot; current distributed file hashes are in [SHA256SUMS.json](../../scripts/config/SHA256SUMS.json).

The October 19 layout revision changes only the two legacy dataset-driver warmup paths in `main/algorithm.jl` and `figure_QSCSD1/algorithm.jl` to the author-created [support input](../../data/Support/README.md). The unified HPR worker continues to warm up on the selected benchmark case. Solver update kernels, stopping rules, selected cases and numerical settings are unchanged.
