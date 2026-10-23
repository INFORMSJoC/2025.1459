# Third-party material

The root MIT license is copied from the authors' HPR-QP repository and applies to their code. It does not relicense datasets, dependencies, the manuscript, or third-party solver material.

Dataset sources, per-family mapping, preparation descriptions, citations and available terms are recorded in [data/README.md](../data/README.md). The same attribution applies to external Dropbox archives. The authors' rights in the specified author-generated numerical inputs are licensed under [CC BY 4.0](../docs/licenses/AUTHOR-DATA-LICENSE.txt); third-party benchmark derivatives retain their upstream terms. The data guide includes OSQP generator credit, the upstream Apache-2.0 license, applicable UCI CC BY 4.0 terms, and the Bodyfat non-commercial notice. Generator-code licenses and dataset terms are recorded separately.

- Julia, CUDA.jl, CUDSS.jl, QPSReader, CSV, DataFrames, MAT, JLD2 and other packages retain their own licenses and are installed through the pinned Julia manifest. NVIDIA CUDA/cuDSS artifacts and the GPU driver remain governed by NVIDIA's terms.
- Saved competitor timings support manuscript table/figure regeneration. The `src/competitors/` folder additionally includes the collected open-source PDQP, SCS.jl and CuClarabel sources, their retained attribution/license files, the required drivers and documented adaptations. These third-party materials retain their original licenses. The external explicit QAP inputs are the authors’ converted relaxations and retain QAPLIB attribution.
- Gurobi requires a valid user installation/license; neither its binary nor license is included. https://www.gurobi.com/

Preserve each dependency’s license. Dataset terms are described separately in the data README; the software license does not grant rights to third-party data.

## UCI data notice

The CC BY 4.0 source credits are: [Abalone](https://doi.org/10.24432/C55C7W), Nash, Sellers, Talbot, Cawthorn and Ford (1994); [Auto MPG](https://doi.org/10.24432/C5859H), Quinlan (1993); and [Qualitative Structure Activity Relationships](https://doi.org/10.24432/C5TP54) (Pyrim/Triazines), Ross King. The prepared inputs apply scaling, feature expansion and Lasso/CQP conversion to data obtained through LIBSVM. Retain these credits and indicate further modifications. See the [CC BY 4.0 license](../docs/licenses/CC-BY-4.0.txt).

## Competing-solver source snapshots

| Solver | Source commit | Bundled location |
|---|---|---|
| CuClarabel | `83b292a00a927419eeb0cb00f05f38e2119efb17` | `src/competitors/CuClarabel` |
| SCS | `6e7bc7d110a6a5fac915943e478de302bffb3be8` | `src/competitors/SCS` |
| PDQP | `135035a55348736b6748dd6511f974db7e615160` | `src/competitors/PDQP` |

Bundled third-party sources retain their upstream licenses. CuClarabel source files are unmodified at the commit listed above. PDQP and SCS adaptations are documented in `src/competitors/patches`.
