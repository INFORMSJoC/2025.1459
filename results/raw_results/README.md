# Raw numerical records

- `<experiment>/<solver>/raw/verification/`: original native JSON and solver logs from the packaged verification runs.
- `<experiment>/<solver>/runs/<attempt>/`: new reviewer runs, created locally; includes settings, launch logs, native output and `record.json`.
- `verification_snapshot.csv`: normalized verification measurements with paths to the native records.
- `reference/initial_release_metadata.json`: original pre-revision release inventory, retained as historical provenance; current file checksums live under `scripts/config`.
- `reference/`: original paper CSV sources, extraction/provenance records and prior validation reports, formerly under `config/reference`.
- `reference/hpr_case_results.json`: historical HPR per-case measurements, separated from the execution settings.
- `reference/archived_results.json`: historical solver rows used as the archived baseline, separated from the experiment catalog.

Native records and historical reference files are retained byte-for-byte during the layout revision. Paths inside those old logs describe the historical run environment; use the current verification index and paper result tables to locate packaged evidence. Shared MM/Primal-Dual records are stored once and referenced by both experiments.

For readable tables and plots, use [paper_results](../paper_results/README.md). The machine-readable inventories under `scripts/config/data` describe inputs and archive parts, not experimental measurements.
