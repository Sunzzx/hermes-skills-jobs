---
name: port-sentinel
description: "Zero-friction local port inspector and zombie dev server terminator. Instantly detects what process is hogging ports (3000, 8000, 8080, 5000, 5173, 8501) and frees them safely."
version: 1.0.0
author: Hermes Autonomous Evolution Engine
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [DevOps, Networking, DeveloperTools, Productivity]
---

# Port Sentinel Skill

## Overview
Eliminates `EADDRINUSE: address already in use` friction during development. Quickly audits listening web servers and kills orphaned instances with a single command.

## CLI Usage
Available globally via `ports`:
```bash
ports check         # Scans ports 3000, 5173, 8000, 8080, 8501, etc.
ports free 3000     # Gracefully terminates the process occupying port 3000
```
