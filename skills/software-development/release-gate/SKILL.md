---
name: release-gate
description: Use when preparing a Git release. Checks local readiness.
version: 1.0.0
author: Hermes Evolution Engine
license: MIT
metadata:
  hermes:
    tags: [git, release, semver, readiness, supply-chain]
    related_skills: [github, artifact-integrity]
---

# Release Gate

## When to Use
Use before tagging or publishing a Git release when you need a deterministic local check for dirty files, missing tracking branches, unpushed commits, or a reused/invalid SemVer tag.

## Overview
Run `release-gate` before creating a version tag or publishing a release. It is a read-only preflight: it never commits, tags, pushes, or changes files.

## Workflow

1. Run from the repository root after tests and build checks have passed.
2. Run `release-gate --tag vMAJOR.MINOR.PATCH` to check repository state and a proposed version.
3. Treat exit code `0` / `ready: true` as a clean release precondition; fix every blocking finding and rerun.
4. Create the tag and publish only after the gate passes; then independently verify remote release state.

```bash
release-gate --tag v2.4.1
release-gate --tag v2.4.1 --format text
```

## Checks

- Current directory is a Git worktree.
- HEAD is on a named branch, not detached.
- No staged, unstaged, or untracked files remain.
- Current branch has an upstream and matches it exactly (neither ahead nor behind).
- Optional tag is valid `vMAJOR.MINOR.PATCH` SemVer and does not already exist locally.

Default output is machine-readable JSON with `ready`, `branch`, `tag`, and per-check `code`, `passed`, and `detail`. Text mode is concise for terminal use. Exit codes: `0` ready, `1` one or more preconditions failed. Git/usage errors are reported as failed checks or argparse errors; do not interpret a failed gate as proof that the release is safe.

## Scope

This gate verifies local Git preconditions only. It does not run tests, inspect package versions, verify signatures, check remote tag collisions, publish artifacts, or create a release. Keep build/test, provenance, and remote publication checks in the release pipeline.

## Common findings

| Finding | Action |
|---|---|
| `dirty_worktree` | Commit or intentionally discard changes; remove stray untracked files. |
| `missing_upstream` | Configure the intended tracking branch and fetch its current state. |
| `upstream_sync` | Push local commits or fetch/rebase/merge upstream changes, then rerun. |
| `invalid_tag` | Use a valid `vMAJOR.MINOR.PATCH` version. |
| `tag_exists` | Select an unused version; never overwrite a published version. |
