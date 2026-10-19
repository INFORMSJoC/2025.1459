# Data used in the paper

One directory per instance type provides an exact instance list, source, preparation description and applicable license notices. No third-party benchmark input files are bundled in this Git snapshot. The previous 40 MM/QAPLIB quick-start files have been removed; benchmark runs require external inputs. The author-defined Appendix example and a tiny initialization-only support input remain bundled.

| Instance type | Selection | Origin |
|---|---:|---|
| [MM](MM/README.md) | 137 | Maros–Mészáros collection; original source linked |
| [Synthetic-CQP](Synthetic-CQP/README.md) | 60 | Author-generated using OSQP benchmark constructions |
| [Lasso-UCI](Lasso-UCI/README.md) | 11 | Third-party LIBSVM regression inputs, expanded/scaled and transformed |
| [Lasso-Synthetic](Lasso-Synthetic/README.md) | 12 | Author-generated using the OSQP Lasso construction |
| [QAPLIB](QAPLIB/README.md) | 36, in MAT and MPS | Third-party QAP cases; authors' convex relaxations |
| [Large-QAP](Large-QAP/README.md) | 6 | Author-generated |
| [Trust-Region](Trust-Region/README.md) | 17 | Author-generated |
| [Appendix-Example](Appendix-Example/README.md) | 1 configuration | Author-defined paper example |
| [Support](Support/README.md) | 1 tiny QP | Initialization only; excluded from benchmark results |

The HPR inventory contains 279 inputs (476.7 GB uncompressed). Competitor QAP runs use 36 additional MPS representations (53.4 GB). The `instances.csv` files list all 315 benchmark representations and their exact SHA-256 hashes. Subset memberships, including Primal-Dual and Restart-Sigma, are recorded by experiment group. The authors' rights in the 95 generated inputs are CC BY 4.0; third-party data keep their original terms.

## Download and extraction

The previously supplied [Dropbox — 2025.1459](https://www.dropbox.com/scl/fo/y0n7luz0yvc192nsk0nwa/ABllkwQ2ZeWGe9HiR6EyJpk?rlkey=y3n3na48lb9f78kpki342dw57&dl=0) contains prepared experiment inputs. **The editor is checking whether Dropbox is acceptable for journal hosting.** This link is retained pending that decision; it does not establish redistribution permission for third-party datasets. The original collection links are provided in the corresponding family directories.

Prepared inputs include paper-specific conversions; original-source downloads are not claimed to match those files without preparation. Use the recorded hashes to verify exact experimental inputs.

### Dropbox folder layout

| Folder | Archive volumes |
|---|---:|
| MM | 1 |
| Synthetic-CQP | 56 |
| Lasso-UCI | 4 |
| Lasso-Synthetic | 74 |
| QAPLIB/HPR-QP | 1 |
| QAPLIB/Competitors | 8 |
| Large-QAP | 4 |
| Trust-Region | 30 |

Files ending in `.partNNN` are consecutive pieces of a compressed archive, not individual instances. Download every part needed by the selected dataset and retain the folder layout. If Dropbox wraps the folder in a ZIP, unpack it first.

From the repository root, extract just MM as follows:

```bash
python3 scripts/manage_data.py extract --groups MM_test_version --cache /path/to/2025.1459 --data-root /path/to/data
python3 scripts/manage_data.py verify --groups MM_test_version --data-root /path/to/data
./scripts/run.sh MM HPR-QP --data-root /path/to/data --plan
```

For every input, add `--include-competitors` and omit `--groups`. Default extraction writes into this `data` directory, with each instance under its family’s `inputs` directory. Downloads and extracted benchmark inputs are excluded from Git and release ZIPs. Existing installations using `MM_mps`, `LASSO`, `QAP`, `RANDOM` and `TR` paths remain accepted by `--data-root` through recorded aliases.

## Why the JSON indexes are retained

Machine-oriented manifests and archive indexes now live under [scripts/config/data](../scripts/config/data/README.md), not among dataset folders. They are required for input identity checks, legacy-path lookup, archive-volume checksums and safe extraction. Readers can use the simpler family-level CSV instance lists. No numerical solver results are stored in these data indexes.
