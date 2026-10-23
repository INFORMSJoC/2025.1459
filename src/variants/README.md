# HPR-QP implementations and provenance

The authors confirm that the code in this directory was developed specifically for the HPR-QP project and has been used only in this project. It is covered by the authors' [MIT license](LICENSE). Julia dependencies and the separate competing-solver snapshots retain their own licenses.

| Directory | Role in the paper |
|---|---|
| `main` | Main dual HPR-QP implementation for explicit CQPs and the explicit Lasso comparison |
| `implicit` | Implicit Lasso and QAP formulations |
| `primal1` | First primal HPR variant |
| `primal2` | Second primal HPR variant |
| `tr_mat` | Second-order-cone trust-region implementation reading the archived MAT inputs |
| `figure_QSCSD1` | Implementation used for the restart/penalty comparison curves |

These are experiment-specific snapshots. Current distributed-file checksums are in [SHA256SUMS.json](../../scripts/config/SHA256SUMS.json).

The legacy dataset-driver helpers in `main` and `figure_QSCSD1` use the author-created [support input](../../data/Support/README.md) for initialization. The unified HPR worker warms up on the selected benchmark case.
