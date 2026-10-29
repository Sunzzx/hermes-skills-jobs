---
name: incident-timeline
description: Build a normalized timeline from mixed application logs.
version: 0.1.0
author: Ayush Kandwal (Sunzzx), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [incident-response, logs, debugging, forensics]
    related_skills: []
---
# Incident Timeline Skill

Merge timestamped service logs into one chronologically ordered, severity-filterable timeline. The bundled CLI uses only Python's standard library and supports ISO-8601, JSON log records, and RFC3164-style syslog timestamps.

## When to Use
- Reconstructing an outage or security incident across multiple services.
- Reviewing large log files without losing original source and line references.
- Producing machine-readable event output for follow-on analysis.

Don't use for untrusted binary log formats, timezone inference, or extracting secrets from logs.

## Prerequisites
- Python 3.9+.
- Installed CLI: `trace-timeline`.

## How to Run
Use `terminal` to run `trace-timeline [--year YYYY] [--min-level LEVEL] [--json] FILE...`. Omit files or pass `-` to read standard input. Naive ISO timestamps are interpreted as UTC; yearless syslog timestamps use current UTC year unless `--year` is supplied.

## Quick Reference
- `trace-timeline api.log worker.log` — merge and sort records.
- `trace-timeline --min-level ERROR --json app.log` — emit only errors and above as JSON.
- `trace-timeline --year 2025 system.log` — assign the explicit year to syslog records.
- `trace-timeline -` — read a log stream from standard input.

## Procedure
1. Gather the relevant log files and record their service/source identity in filenames; each output event retains source path and original line number.
2. Run the CLI across all sources, choosing `--year` when syslog timestamps omit it. Completion: every parsed record has a normalized UTC timestamp.
3. Review skipped-line count on stderr. Resolve missing or malformed timestamps at source, rather than treating dropped lines as incident evidence.
4. Narrow with `--min-level` only after the full timeline is captured. Completion: preserve an unfiltered copy for auditability.
5. Use `--json` for downstream analysis; output events remain sorted by normalized timestamp, source, and original line.

## Pitfalls
- RFC3164 syslog does not carry a year or timezone; set `--year` and interpret the UTC assumption deliberately.
- Unknown severity labels remain visible in the default output but are excluded by severity filters.
- Mixed logs with incompatible timestamp conventions require a preprocessing step; the CLI skips unparseable records and reports the count.
- Avoid sending logs containing credentials or personal data to external services; this tool processes files locally.

## Verification
Run the installed `trace-timeline --help`. For behavioral tests, run `python3 tests/test_trace_timeline.py` from the repository root; tests cover sorting, timezone conversion, JSON records, syslog year assignment, filtering, and malformed input.
