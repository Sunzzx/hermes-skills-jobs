---
name: artifact-integrity
description: Create and verify SHA-256 manifests for release artifacts.
version: 0.1.0
author: Ayush Kandwal (Sunzzx), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [release-integrity, sha256, supply-chain, artifact-audit]
    related_skills: []
---
# Artifact Integrity Skill

Create a deterministic local inventory of release files and verify it before distribution or after transfer. The standard-library CLI records SHA-256 and byte length, rejects symlink files, and reports missing, changed, and unexpected files without modifying the audited tree.

## When to Use
- Auditing build outputs, downloaded bundles, or handoffs before release.
- Checking that an archive extraction or file transfer preserved all inventoried files.
- Producing a machine-readable integrity record for later verification.

Don't use this as a digital signature: an attacker who can replace both files and manifest can forge the pair. Sign or publish the manifest through a separately trusted channel for authenticity.

## Prerequisites
- Python 3.9+; no third-party packages.
- Installed CLI: `artifact-integrity`.

## How to Run
Use `terminal` to run `artifact-integrity create ROOT --output MANIFEST.json [--exclude GLOB]` and later `artifact-integrity verify ROOT MANIFEST.json`. Pass identical repeated `--exclude` patterns to both commands.

## Quick Reference
- `artifact-integrity create dist --output dist-manifest.json` — inventory all regular files.
- `artifact-integrity create dist -o release.json --exclude '*.map' --exclude 'debug/**'` — omit matching relative paths.
- `artifact-integrity verify dist release.json` — verify exact inventory; unexpected files fail too.
- Exit status: `0` verified/created, `1` mismatch, `2` invalid input or I/O failure.

## Procedure
1. Build the release tree and choose exclusions deliberately; avoid excluding runtime-critical files. Completion: the exact intended file set is documented.
2. Create the manifest outside the tree being inventoried. Completion: JSON reports the expected file count and is stored with the release record.
3. Transfer or stage the artifact and manifest independently as needed.
4. Verify the destination tree using the same exclusions. Completion: output begins `OK:` and process status is `0`; resolve all mismatch lines before release.
5. Preserve the manifest in a trusted, access-controlled location if it is evidence of the expected release contents.

## Pitfalls
- A manifest alone does not authenticate its creator; use a detached signature or trusted immutable release channel when provenance matters.
- Symlinked files are rejected rather than followed. Symlinked directories are not traversed, so their contents are outside the inventory.
- Glob patterns match relative POSIX-style paths; patterns are case-sensitive and should be repeated exactly for verification.
- Any newly added file is reported as unexpected, by design.

## Verification
Run `artifact-integrity --help` and `python3 tests/test_artifact_integrity.py` from the repository root. Tests exercise clean verification, tampering, missing/extra files, exclusions, and symlink rejection.
