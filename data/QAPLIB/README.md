# QAPLIB instances

**Origin:** Third-party benchmark, relaxations prepared by authors. **Selection:** 36 cases, each in MAT and MPS. The exact files, hashes, sizes and experiment-group membership are listed in [instances.csv](instances.csv).

[Original source / construction reference](https://coral.ise.lehigh.edu/data-sets/qaplib/). Cite Burkard, Karisch and Rendl (1997).

The 36 selected cases exclude sko90. MAT and MPS are representations of the same cases, not additional datasets. HPR-QP uses MAT; competitors use MPS.

## Terms

No explicit redistribution license was identified on the reviewed QAPLIB pages. No QAPLIB input files are included in this Git snapshot. Link to the original collection; do not apply the authors’ generated-data license to these benchmarks.

No blanket dataset license is asserted. See also [third-party notices](../../docs/THIRD_PARTY.md).

## Acquisition and layout

Prepared archive location on Dropbox: `QAPLIB/HPR-QP/ (1 part); QAPLIB/Competitors/ (8 parts)`. See the [data guide](../README.md) for download and extraction instructions. The original source is provided above where applicable; original downloads may need the paper’s transformations and are not asserted to match prepared MPS/MAT hashes directly.

Extracted path beneath `--data-root`: `QAPLIB/inputs/implicit/*.mat; QAPLIB/inputs/explicit/*.mps`.

From the repository root:

```bash
python3 scripts/manage_data.py extract --groups QAPLIB --cache /path/to/2025.1459 --data-root /path/to/data
python3 scripts/manage_data.py verify --groups QAPLIB --data-root /path/to/data
```

For competitors, add `--include-competitors --groups QAPLIB_competitors` to the extraction command instead. Download all eight competitor archive parts. For example:

```bash
python3 scripts/manage_data.py extract --include-competitors --groups QAPLIB_competitors --cache /path/to/2025.1459 --data-root /path/to/data
```
