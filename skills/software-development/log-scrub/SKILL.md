---
name: log-scrub
description: Redact common credentials and personal identifiers from logs.
version: 1.0.0
author: Ayush Kandwal (Sunzzx), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [LogSanitization, Secrets, Privacy, IncidentResponse]
    related_skills: [incident-timeline]
---

# Log Scrub

Sanitize text logs before sharing or attaching them to an issue. The standard-library CLI redacts common token formats, private-key blocks, email addresses, and phone-like strings; it never edits input files in place. Pattern matching is heuristic, so treat the output as a first-pass filter rather than proof that a document contains no secrets.

## When to Use
- Remove common credentials or personal identifiers from logs before sharing.
- Run a content-free check to detect likely sensitive strings.
- Don't use as a substitute for secret scanning of source repositories or binary files.

## Prerequisites
- Python 3.10+; no third-party packages.
- CLI installed as `log-scrub` in `~/.local/bin/`.

## How to Run
```text
log-scrub app.log > app-sanitized.log
log-scrub --check app.log
cat app.log | log-scrub -
```
`--check` writes category counts only to stderr, emits no source content, and exits 1 when matches are detected, 0 when clean, or 2 on input errors. Normal mode writes sanitized content to stdout and a count summary to stderr.

## Procedure
1. Preserve the source log and choose a separate destination for sanitized output; the tool never overwrites source files.
2. Run `log-scrub --check <file>`; a non-zero result indicates likely sensitive content.
3. Generate output with `log-scrub <file> > <sanitized-file>` and inspect the sanitized version for context and missed formats.
4. Share only the reviewed output; completion means the output has no known raw match from the supported patterns and was manually checked.

## Pitfalls
- Detection is pattern-based and may miss custom secrets, encoded values, binary formats, or unusual token structures.
- Phone matching is intentionally broad and can redact dates or numeric identifiers; inspect surrounding context.
- Multi-file input is concatenated in argument order; preserve boundaries yourself if needed.

## Verification
Use `log-scrub --check` to detect known patterns without exposing text. Confirm the result with tests in `tests/test_log_scrub.py` and a manual review of the output before sharing.
