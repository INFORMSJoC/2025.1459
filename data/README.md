# Data layout and acquisition

**Download datasets:** [Dropbox — 2025.1459](https://www.dropbox.com/scl/fo/y0n7luz0yvc192nsk0nwa/ABllkwQ2ZeWGe9HiR6EyJpk?rlkey=y3n3na48lb9f78kpki342dw57&dl=0).

The package bundles 40 compact inputs (7.36 MB): all 36 QAPLIB MAT files and the MM files `AUG2D.mps`, `CVXQP3_S.mps`, `QSCSD1.mps` and `QSCSD6.mps`. Full QAPLIB HPR-QP and Restart-Sigma runs need no external data. Other full experiments require their corresponding downloads below. Appendix-Example uses a bundled JSON configuration.

The HPR input inventory contains 279 files (476.7 GB uncompressed). QAPLIB competitors additionally use 36 explicit MPS files (53.4 GB). Input sizes and SHA-256 hashes are listed in `manifest.json` and `competitor_qap_manifest.json`.

## Canonical paths used by each experiment

Pass `--data-root` the directory **containing** the following folders. Names and capitalization are significant on Linux.

| Experiment | Path below the data root | Solvers |
|---|---|---|
| MM, Primal-Dual, Restart-Sigma | `MM_mps/*.mps` | All applicable solvers |
| Synthetic-CQP | `RANDOM/new/*.mps`, `RANDOM/denser/*.mps` | All applicable solvers |
| Lasso-UCI | `LASSO/uci/*.mps` | Implicit/explicit HPR-QP and competitors |
| Lasso-Synthetic | `LASSO/synthetic/*.mps` | Implicit/explicit HPR-QP and competitors |
| QAPLIB | `QAP/mat/new/*.mat` | HPR-QP |
| QAPLIB | `QAP/mps/*.mps` | PDQP, SCS, CuClarabel, Gurobi |
| Large-QAP | `QAP/mat_random/*.mat`, `QAP/mat_random/large/*.mat` | HPR-QP |
| Trust-Region | `TR/mat/*.mat`, `TR/mat/large/*.mat` | All applicable solvers |
| Appendix-Example | Package-local `config/appendix_example.json` | HPR-QP, HPR-p1; no external input |

Set `HPR_DATA_ROOT` or pass `--data-root /path/to/QP_data`. Bundled inputs remain available as a fallback. Relative command-line paths are resolved from the directory where the command is invoked; quote paths containing spaces. Outputs default to the package's `experiments/` directory; `--output` selects another location.

```bash
export HPR_DATA_ROOT=/path/to/QP_data
./run.sh ALL ALL --plan             # List selected cases and missing inputs; no solves
./run.sh Lasso-UCI HPR-QP
```

Synthetic-CQP contains two groups whose filenames can overlap. For individual-case selection, use `--groups synthetic_paper` or `--groups revision_denser`, or the exact case key shown by `--plan`. An unqualified matching basename selects both groups.

## Dropbox folder layout

| Folder beneath `2025.1459/` | Archive volumes | Contents |
|---|---:|---|
| `MM/` | 1 | MM inputs |
| `Synthetic-CQP/` | 56 | 30 original and 30 denser synthetic CQPs |
| `Lasso-UCI/` | 4 | UCI Lasso inputs |
| `Lasso-Synthetic/` | 74 | Synthetic Lasso inputs |
| `QAPLIB/HPR-QP/` | 1 | Compact MAT inputs |
| `QAPLIB/Competitors/` | 8 | Explicit MPS inputs |
| `Large-QAP/` | 4 | Random large QAP inputs |
| `Trust-Region/` | 30 | All 17 instances, dimensions 10,000–20,000,000 |

Trust-Region contains all 17 instances in one experiment folder; Synthetic-CQP contains both the original and denser sets in one folder. QAPLIB's two subfolders contain different input formats for the **same 36 cases**: compact MAT inputs for implicit HPR-QP, and explicit MPS inputs for PDQP, SCS, CuClarabel and Gurobi. The launcher chooses the format from the solver argument.

Files ending in `.partNNN` are consecutive parts of compressed dataset archives, **not individual instances**. Download every part of each required archive. Preserve the filenames and folder layout above. If Dropbox provides a ZIP for a folder download, unpack that ZIP first. Set `--cache` to the downloaded `2025.1459` directory. The extraction script joins the parts, verifies their hashes and writes the canonical input paths automatically.

## Download and extract

For the complete MM or Primal-Dual experiment, download `MM/MM_mps.tar.gz.part000` (about 37 MB), then run from the package root:

```bash
python3 scripts/manage_data.py extract --groups MM_test_version --cache /path/to/2025.1459 --data-root /path/to/QP_data
./run.sh Primal-Dual HPR-QP --data-root /path/to/QP_data
```

For a complete installation, download all experiment folders and run:

```bash
python3 scripts/manage_data.py extract --include-competitors --cache /path/to/2025.1459 --data-root /path/to/QP_data
python3 scripts/manage_data.py verify --include-competitors --data-root /path/to/QP_data
export HPR_DATA_ROOT=/path/to/QP_data
./run.sh ALL ALL --plan
```

For only the QAPLIB competitor inputs, download all eight parts under `QAPLIB/Competitors/` and run:

```bash
python3 scripts/manage_data.py extract --include-competitors --groups QAPLIB_competitors --cache /path/to/2025.1459 --data-root /path/to/QP_data
./run.sh QAPLIB SCS --data-root /path/to/QP_data
```

Use `python3 scripts/manage_data.py extract --help` for group-selection options. The archive indexes in `archives/` contain the expected filenames, sizes and SHA-256 hashes for all 178 parts (182.5 GB compressed). Extraction verifies the archives and selected inputs. The Dropbox folder also contains `SHA256SUMS.txt`. The supported download workflow is to download the required folders through Dropbox and then run `extract` as above.

## Dataset attribution

These credits and terms apply to the bundled inputs and the corresponding external Dropbox archives. Paths refer to the data root. The manifests identify each prepared file by name, size and SHA-256. Prepared MPS/MAT files include the transformations below; they are not the original upstream downloads. The root MIT license covers the authors' software and does not replace third-party data or generator-code terms.

| Experiment / prepared inputs | Files | Source and preparation | Attribution / terms |
|---|---:|---|---|
| MM; `MM_mps/` (also used by Primal-Dual and Restart-Sigma) | 137 | [Maros–Mészáros collection](https://www.doc.ic.ac.uk/~im/00README.QP); prepared MPS/QPS inputs, excluding VALUES | Maros and Mészáros (1999). No explicit redistribution license identified in the collection README. |
| Synthetic-CQP; `RANDOM/new/`, `RANDOM/denser/` | 60 | Authors' instances based on/adapted from [OSQP benchmark generators](https://github.com/osqp/osqp_benchmarks/tree/master/problem_classes): equality-constrained QP, random QP, control, Huber, portfolio and SVM; dimensions/sparsity selected for this paper, exported to MPS | Stellato et al. (2020); OSQP generator code: Apache-2.0, see notice below. |
| Lasso-UCI; `LASSO/uci/` | 11 | [LIBSVM regression datasets](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html), using the feature-expanded/scaled inputs studied by Li, Sun and Toh (2018), followed by the paper's Lasso/CQP formulation | Chang and Lin (2011); Li, Sun and Toh (2018); source terms below. |
| Lasso-Synthetic; `LASSO/synthetic/` | 12 | Authors' generated sparse Lasso instances following Appendix A.5 of Stellato et al. (2020), with paper-specific dimensions and sparsity, exported to MPS | Credit the HPR-QP paper and OSQP; generator-code attribution below. |
| QAPLIB; `QAP/mat/new/` and `QAP/mps/` | 36 MAT + 36 MPS | Authors' convex relaxations of the same 36 [QAPLIB](https://coral.ise.lehigh.edu/data-sets/qaplib/) cases; compact MAT for implicit HPR-QP and explicit MPS for competitors; excludes sko90 | Burkard, Karisch and Rendl (1997). No explicit redistribution license identified on the reviewed QAPLIB pages. |
| Large-QAP; `QAP/mat_random/` including `large/` | 6 | Authors' randomly generated QAP matrices and convex-relaxation inputs | Cite the HPR-QP paper. |
| Trust-Region; `TR/mat/` including `large/` | 17 | Authors' random trust-region instances converted to the convex SOCP formulation described in the paper | Cite the HPR-QP paper for the instances and Ho-Nguyen and Kılınç-Karzan (2017) for the formulation. |
| Appendix-Example; `config/appendix_example.json` | 1 configuration | Numerical example specified in the HPR-QP paper | Cite the HPR-QP paper. |

The inventory comprises **315 input files**: 279 HPR inputs and 36 additional explicit-QAP MPS inputs. The appendix configuration is separate. Use the supplied archives for exact experimental inputs.

### Author-generated data license

The rights controlled by the HPR-QP authors in the 95 synthetic-CQP, synthetic-Lasso, random-QAP and trust-region numerical inputs are licensed under **CC BY 4.0**. The [license declaration](licenses/AUTHOR-DATA-LICENSE.txt) identifies the covered paths and attribution; the [full license text](licenses/CC-BY-4.0.txt) is included. This applies to the corresponding external archives as well. Third-party benchmark data and OSQP-derived generator code retain their own terms.

### OSQP generator credit

The synthetic benchmark generation draws on **osqp/osqp_benchmarks**, licensed under **Apache License 2.0**. The upstream source includes `random_qp.py`, `eq_qp.py`, `control.py`, `huber.py`, `portfolio.py`, `svm.py` and `lasso.py`. Credit both the [benchmark repository](https://github.com/osqp/osqp_benchmarks) and Stellato et al. (2020) when reusing these constructions. A copy of the upstream license, including its copyright notice, is included as [licenses/OSQP-Apache-2.0.txt](licenses/OSQP-Apache-2.0.txt).

The HPR-QP instances use paper-specific sizes, sparsities and formulations; they are not the unmodified OSQP benchmark release. The reproduction workflow loads the prepared inputs and does not require executing OSQP's generators. The Apache notice applies to the upstream generator code and any code adaptations; it is not a blanket license declaration for every numerical output or third-party benchmark dataset.

### Trust-region construction

The formulation reference is Ho-Nguyen and Kılınç-Karzan (2017), as cited in the HPR-QP paper. The random instances and MATLAB generator are the authors' work.

The trust-region MAT inputs use the authors' MATLAB construction: draw a sparse symmetric matrix with `sprandsym(d, 50/d)`, estimate its smallest eigenvalue with `eigs(h, 1, 'SA')`, and shift its diagonal using 1.01 times a nonpositive estimate (or 0.99 times a positive estimate). Embed the shifted matrix as `H = blkdiag(0, h - shift*I)`, set `c = [0; randn(d,1)]`, and impose `A = [1, 0, ..., 0]`, `b = 1`. The solver uses the unit second-order cone for the augmented variable. The saved variables `H`, `c`, `A`, `b` are in MATLAB v7.3 format. This construction is separate from the OSQP synthetic-CQP generators. The 17 supplied inputs cover dimensions 10,000 through 20,000,000; use their archived bytes for the paper experiments.

### Lasso-UCI source terms

Inputs come from the [LIBSVM regression collection](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html), with scaling, feature expansion and the paper's Lasso/CQP conversion. Cite Chang and Lin (2011) and Li, Sun and Toh (2018). Applicable UCI data retain [CC BY 4.0](licenses/CC-BY-4.0.txt) and the [source credits](../docs/THIRD_PARTY.md#uci-data-notice). Bodyfat retains its [non-commercial notice](licenses/Bodyfat-NOTICE.txt). The [10-K Corpus](https://www.cs.cmu.edu/~ark/10K/) requests academic-research use and citation to Kogan et al. (2009). Other source data retain their upstream terms; this package does not assign them a new license.

### References

- Ho-Nguyen, N., and F. Kılınç-Karzan (2017). A second-order cone based approach for solving the trust region subproblem and its variants. [Paper](https://arxiv.org/abs/1603.03366), DOI [10.1137/16M1065197](https://doi.org/10.1137/16M1065197).
- Maros, I., and C. Mészáros (1999). A repository of convex quadratic programming problems. *Optimization Methods and Software*, 11–12, 671–681.
- Burkard, R. E., S. E. Karisch, and F. Rendl (1997). QAPLIB—A quadratic assignment problem library. *Journal of Global Optimization*, 10, 391–403.
- Chang, C.-C., and C.-J. Lin (2011). LIBSVM: A library for support vector machines. *ACM Transactions on Intelligent Systems and Technology*, 2(3), Article 27.
- Li, X., D. Sun, and K.-C. Toh (2018). A highly efficient semismooth Newton augmented Lagrangian method for solving Lasso problems. *SIAM Journal on Optimization*, 28(1), 433–458.
- Kogan, S., D. Levin, B. R. Routledge, J. S. Sagi, and N. A. Smith (2009). Predicting risk from financial reports with regression. *NAACL-HLT*, 272–280.
- Penrose, K. W., A. G. Nelson, and A. G. Fisher (1985). Generalized body composition prediction equation for men using simple measurement techniques. *Medicine and Science in Sports and Exercise*, 17(2), 189 (abstract).
- Harrison, D., and D. L. Rubinfeld (1978). Hedonic prices and the demand for clean air. *Journal of Environmental Economics and Management*, 5, 81–102.
- Stellato, B., G. Banjac, P. Goulart, A. Bemporad, and S. Boyd (2020). OSQP: An operator splitting solver for quadratic programs. *Mathematical Programming Computation*, 12, 637–672. [Benchmark code](https://github.com/osqp/osqp_benchmarks).
