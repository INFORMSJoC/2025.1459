# Trust-Region instances

**Origin:** Author-generated. **Selection:** 17. The exact files, hashes, sizes and experiment-group membership are listed in [instances.csv](instances.csv).

[Original source / construction reference](https://doi.org/10.1137/16M1065197). Cite The HPR-QP paper; Ho-Nguyen and Kılınç-Karzan (2017).

MAT inputs cover dimensions 10,000–20,000,000. The authors’ MATLAB construction uses sprandsym, an estimated eigenvalue shift (with the recorded 1% adjustment), and randn before the SOCP embedding. It is separate from OSQP’s generators; no QPALM/QSDPNAL generator provenance is claimed.

## Terms

The authors’ rights in these generated trust-region inputs are CC BY 4.0.

[LICENSE](LICENSE), [CC-BY-4.0.txt](CC-BY-4.0.txt) See also [third-party notices](../../docs/THIRD_PARTY.md).

## Acquisition and layout

Prepared archive location on Dropbox: `Trust-Region/ (30 parts)`. See the [data guide](../README.md) for download and extraction instructions. The original source is provided above where applicable; original downloads may need the paper’s transformations and are not asserted to match prepared MPS/MAT hashes directly.

Extracted path beneath `--data-root`: `Trust-Region/inputs/*.mat and large/*.mat`.

From the repository root:

```bash
python3 scripts/manage_data.py extract --groups trust_region trust_region_large --cache /path/to/2025.1459 --data-root /path/to/data
python3 scripts/manage_data.py verify --groups trust_region trust_region_large --data-root /path/to/data
```
