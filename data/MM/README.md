# MM instances

**Origin:** Third-party. **Selection:** 137. The exact files, hashes, sizes and experiment-group membership are listed in [instances.csv](instances.csv).

[Original source / construction reference](https://www.doc.ic.ac.uk/~im/00README.QP). Cite Maros and Mészáros (1999).

The selected set excludes VALUES. Primal-Dual uses the 30 cases identified by the `primal1`/`primal2` groups; Restart-Sigma reuses QSCSD1.

## Terms

The collection README does not provide an explicit redistribution license. No MM instance files are included in this Git snapshot. Use the original source link; the prepared archive preserves the paper input bytes.

No blanket dataset license is asserted. See also [third-party notices](../../docs/THIRD_PARTY.md).

## Acquisition and layout

Prepared archive location in the existing Dropbox collection: `MM/MM_mps.tar.gz.part000`. Dropbox hosting is under editorial consideration; see the [data guide](../README.md). The original source is provided above where applicable; original downloads may need the paper’s transformations and are not asserted to match prepared MPS/MAT hashes directly.

Extracted path beneath `--data-root`: `MM/inputs/*.mps`.

From the repository root:

```bash
python3 scripts/manage_data.py extract --groups MM --cache /path/to/2025.1459 --data-root /path/to/data
python3 scripts/manage_data.py verify --groups MM --data-root /path/to/data
```
