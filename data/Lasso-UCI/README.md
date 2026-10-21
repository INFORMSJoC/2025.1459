# Lasso-UCI instances

**Origin:** Third-party, transformed by authors. **Selection:** 11. The exact files, hashes, sizes and experiment-group membership are listed in [instances.csv](instances.csv).

[Original source / construction reference](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html). Cite Chang and Lin (2011); Li, Sun and Toh (2018).

Feature-expanded/scaled inputs are converted to the paper’s Lasso formulation; the supplied prepared archives are required to match the listed hashes exactly.

## Terms

No third-party instance files are bundled. Applicable UCI data retain CC BY 4.0; Bodyfat retains its non-commercial notice. These terms do not cover all LIBSVM datasets. Source links and required credits are retained in the third-party notices.

[CC-BY-4.0.txt](CC-BY-4.0.txt), [Bodyfat-NOTICE.txt](Bodyfat-NOTICE.txt) See also [third-party notices](../../docs/THIRD_PARTY.md).

## Acquisition and layout

Prepared archive location on Dropbox: `Lasso-UCI/ (4 parts)`. See the [data guide](../README.md) for download and extraction instructions. The original source is provided above where applicable; original downloads may need the paper’s transformations and are not asserted to match prepared MPS/MAT hashes directly.

Extracted path beneath `--data-root`: `Lasso-UCI/inputs/*.mps`.

From the repository root:

```bash
python3 scripts/manage_data.py extract --groups LASSO_UCI --cache /path/to/2025.1459 --data-root /path/to/data
python3 scripts/manage_data.py verify --groups LASSO_UCI --data-root /path/to/data
```
