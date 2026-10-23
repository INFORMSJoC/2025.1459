# Machine-readable data inventories

Readers should start with the per-family [data guide](../../../data/README.md) and `instances.csv` lists. These JSON files remain because the launcher and extraction checks consume them:

- `manifest.json`: exact filenames, byte sizes, SHA-256 hashes and group membership for the 279 HPR inputs; `legacy_paths` accepts existing installations.
- `competitor_qap_manifest.json`: the 36 explicit MPS representations of the QAPLIB cases.
- `archives/archive_index.json`: archive volumes and the mapping from historical archive member paths to the current per-family paths.
- `archives/competitor_qap_index.json`: the additional explicit-QAP archive volumes.

The archive indexes describe the prepared archives hosted on Dropbox. They are not experimental results and are not data licenses. No third-party benchmark inputs are bundled in the source snapshot. The `bundled` flag is false for all 315 benchmark representations.
