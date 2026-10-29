---
name: jsonl-audit
description: "Validate JSONL records and report schema drift."
version: 1.0.0
author: Ayush Kandwal (Sunzzx), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [JSONL, DataQuality, SchemaValidation, ETL]
    related_skills: [incident-timeline]
---

# JSONL Audit

Check newline-delimited JSON datasets for invalid records, duplicate object keys, schema inconsistency, type drift, missing identifiers, and duplicate identifier values. It streams input and emits a machine-readable summary; it does not modify source data or infer a business schema.

## When to Use
- Validate JSONL before ingestion, model training, or committing fixture data.
- Inspect field presence and per-field type distributions without loading the full dataset into memory.
- Don't use as a substitute for domain-specific schema validation or privacy scanning.

## Prerequisites
- Python 3.10+; standard library only.
- CLI installed as `jsonl-audit` in `~/.local/bin/`.

## How to Run
```text
jsonl-audit records.jsonl
jsonl-audit --strict-fields records.jsonl
jsonl-audit --unique record_id records.jsonl
jsonl-audit --unique record_id < records.jsonl
```

The report is JSON on stdout. Exit codes: `0` means no findings, `1` means data-quality findings, and `2` means an input or decoding error. Blank lines are ignored; each non-blank line must decode to one JSON object. The unique key is top-level and values are compared by canonical JSON representation.

## Procedure
1. Run `jsonl-audit <file>` to inspect parse validity and per-field type/presence counts; completion criterion is a report with no parsing errors.
2. Add `--strict-fields` when all rows must share the first valid record's key set; completion criterion is no row-level field-set mismatches.
3. Add `--unique <key>` when the dataset has a stable top-level identifier; completion criterion is no missing or repeated key findings.
4. Review the schema report against the intended domain contract; a clean audit only confirms these structural checks.

## Pitfalls
- Type counts are observational, not validation rules; mixed types are reported but not rejected by default.
- Strict-field comparison uses the first valid object as a baseline, so choose representative input ordering or use domain validation.
- Blank lines are ignored and do not count as records.
- JSONL object rows only; arrays and scalar values are reported as invalid rows.

## Verification
Run `python3 -m unittest discover -s tests -p 'test_jsonl_audit.py' -v`. Exercise `jsonl-audit --unique id --strict-fields <file>` and check the JSON `errors` array and exit code before accepting data.
