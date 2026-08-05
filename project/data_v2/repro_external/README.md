# Reproducible external sequence datasets

This directory contains the compact, non-binary inputs needed to reconstruct the
three real-source external evaluations. Large upstream repositories, nested Git
metadata, executables, payloads, certificates, and model artifacts are excluded.

## Contents

- `attack_flow/corpus/`: 40 source `.afb` files and the upstream Apache-2.0
  license. The current loader excludes the two CTID-overlapping Turla flows.
- `ctid/raw_yaml/`: the 10 emulation-plan YAML files used by the parser.
- `ctid/parsed/`: the 20 CSV inputs read by `load_ctid_external`.
- `stockpile/data/`: only the ability and adversary-profile YAML files read by
  the loader; payloads and binaries are deliberately omitted.
- `loader_snapshots/`: deterministic exports of what the current loaders return.
- `MANIFEST.json`: source revisions, row counts, prefix-length distributions,
  file sizes, and SHA-256 hashes.

Run the export from the repository root:

```powershell
python project/data_v2/scripts/export_repro_external.py
```

## Important sequence warning

The loader snapshots are audit artifacts, not yet the final semantic-branch
dataset. The current Attack Flow and Stockpile loaders encode every example as a
single source technique followed by one target technique. Consequently, all 705
Attack Flow rows and all 65 Stockpile rows have `prefix_len=1`. CTID has 281
cumulative-prefix rows with lengths from 1 to 71.

Before LLM-reasoning generation, Stockpile should be rebuilt with cumulative
profile prefixes and Attack Flow needs a frozen DAG-to-prefix policy. The current
snapshots are retained so the previously reported external results remain
reproducible and the change in data semantics is explicit.

## Provenance

The exact upstream revisions are recorded in `MANIFEST.json`. All three upstream
projects are distributed under Apache License 2.0; their license text is copied
into the corresponding source directory.
