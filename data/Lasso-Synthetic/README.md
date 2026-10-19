# Lasso-Synthetic instances

**Origin:** Author-generated. **Selection:** 12. The exact files, hashes, sizes and experiment-group membership are listed in [instances.csv](instances.csv).

[Original source / construction reference](https://github.com/osqp/osqp_benchmarks). Cite Stellato et al. (2020), and the HPR-QP paper.

Paper-specific dimensions and sparsity, converted to MPS; implicit and explicit HPR comparisons use the same input files.

## Terms

The authors’ rights in these generated inputs are CC BY 4.0. The OSQP Lasso construction and adapted generator code are credited separately under Apache-2.0.

[LICENSE](LICENSE), [CC-BY-4.0.txt](CC-BY-4.0.txt), [OSQP-LICENSE](OSQP-LICENSE) See also [third-party notices](../../docs/THIRD_PARTY.md).

## Acquisition and layout

Prepared archive location in the existing Dropbox collection: `Lasso-Synthetic/ (74 parts)`. Dropbox hosting is under editorial consideration; see the [data guide](../README.md). The original source is provided above where applicable; original downloads may need the paper’s transformations and are not asserted to match prepared MPS/MAT hashes directly.

Extracted path beneath `--data-root`: `Lasso-Synthetic/inputs/*.mps`.

From the repository root:

```bash
python3 scripts/manage_data.py extract --groups LASSO_synthetic --cache /path/to/2025.1459 --data-root /path/to/data
python3 scripts/manage_data.py verify --groups LASSO_synthetic --data-root /path/to/data
```
