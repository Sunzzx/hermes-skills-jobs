---
name: env-guard
description: Use when auditing .env files. Enforces drift prevention, secret leak blocking, and schema validation.
version: 1.0.0
author: Hermes Evolution Engine
license: MIT
metadata:
  hermes:
    tags: [devops, environment, secrets, drift, validation, developer-tools]
    related_skills: [log-scrub, release-gate, artifact-integrity]
---

# Env Guard

## When to Use
Use before committing code, running CI test suites, or deploying applications to verify that local environment configurations (`.env`) match public templates (`.env.example`), detect secret leaks in committed templates, and enforce strict type/presence schemas.

## Overview
`env-guard` is an autonomous drift, secret leak, and schema enforcement engine for environment files. It inspects `.env` and `.env.example` configurations, alerts on missing or undocumented keys, checks for accidental token/key leaks in templates, and validates type constraints without external dependencies.

## Workflow

1. **Audit Environment Drift**:
   Run `env-guard` in any project root to compare `.env` against `.env.example`.
   ```bash
   env-guard
   env-guard --format json
   ```
2. **Enforce Strict Undocumented Key Prevention**:
   Ensure team members do not introduce undeclared environment variables that would break production deployments.
   ```bash
   env-guard --strict
   ```
3. **Validate Required Keys & Types in CI**:
   Enforce required variables and type assertions before test suites run.
   ```bash
   env-guard --required DATABASE_URL,PORT,SECRET_KEY --types PORT:port,DEBUG:bool,APP_URL:url
   ```
4. **Safely Sync Templates**:
   Scaffold missing keys from `.env` into `.env.example` using dummy placeholders, ensuring actual secrets are never copied.
   ```bash
   env-guard --sync
   ```

## Checks & Enforcement

- **Drift Detection**: Identifies keys declared in `.env.example` that are missing in `.env`.
- **Strict Cleanliness**: Detects undocumented variables in `.env` absent from `.env.example`.
- **Template Secret Scanner**: Catches active AWS credentials, GitHub PATs, private keys, JWTs, and OpenAI/Stripe tokens accidentally placed into public `.env.example` files.
- **Type Checking**: Validates integer, port (1-65535), boolean (`true`/`false`/`1`/`0`), URL (`http://`, `postgres://`, etc.), email, and JSON string types.
- **Safe Template Synchronization**: Preserves existing comments and appends safe placeholders (`your_<key>_here`).

## Common Findings & Remediation

| Finding | Meaning | Action |
| :--- | :--- | :--- |
| `missing_in_env` | Key declared in example is missing in `.env`. | Add key with local value to `.env`. |
| `extra_in_env` | Undocumented key present in `.env` during `--strict`. | Document key in `.env.example` or remove if unused. |
| `secret_leaks` | Live API key or token found in `.env.example`. | **CRITICAL**: Revoke key immediately and replace with placeholder in template. |
| `schema_violations` | Variable failed type constraint (e.g. invalid port number). | Correct value to match expected type specification. |
| `empty_values` | Required variable defined with empty string. | Provide valid non-empty configuration value. |
