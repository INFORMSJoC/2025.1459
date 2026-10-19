# Synthetic-CQP instances

**Origin:** Author-generated. **Selection:** 60. The exact files, hashes, sizes and experiment-group membership are listed in [instances.csv](instances.csv).

[Original source / construction reference](https://github.com/osqp/osqp_benchmarks). Cite Stellato et al. (2020), and the HPR-QP paper.

30 original and 30 denser instances; repeated basenames distinguish different inputs by subdirectory/group.

## Terms

The authors’ rights in these generated inputs are CC BY 4.0. OSQP-derived generator code retains Apache-2.0; both texts are included locally.

[LICENSE](LICENSE), [CC-BY-4.0.txt](CC-BY-4.0.txt), [OSQP-LICENSE](OSQP-LICENSE) See also [third-party notices](../../docs/THIRD_PARTY.md).

## Acquisition and layout

Prepared archive location in the existing Dropbox collection: `Synthetic-CQP/ (56 parts)`. Dropbox hosting is under editorial consideration; see the [data guide](../README.md). The original source is provided above where applicable; original downloads may need the paper’s transformations and are not asserted to match prepared MPS/MAT hashes directly.

Extracted path beneath `--data-root`: `Synthetic-CQP/inputs/{original,denser}/*.mps`.

From the repository root:

```bash
python3 scripts/manage_data.py extract --groups synthetic_paper revision_denser --cache /path/to/2025.1459 --data-root /path/to/data
python3 scripts/manage_data.py verify --groups synthetic_paper revision_denser --data-root /path/to/data
```
