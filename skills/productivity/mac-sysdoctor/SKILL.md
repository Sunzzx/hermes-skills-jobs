---
name: mac-sysdoctor
description: "Autonomous macOS performance doctor: inspects memory pressure, detects rogue CPU hogging processes, purges dead caches, audits thermal throttling, and optimizes battery life."
version: 1.0.0
author: Hermes Autonomous Evolution Engine
license: MIT
platforms: [macos]
metadata:
  hermes:
    tags: [macOS, Performance, SystemOptimization, Hardware, Productivity]
---

# Mac SysDoctor Skill

## Overview
Provides immediate diagnostic and optimization commands for macOS system health, thermal state, memory pressure, and background zombie processes.

## CLI Usage
Available globally via `sysdoctor`:
```bash
sysdoctor status    # Full health check: memory, CPU hogs, disk, battery
sysdoctor optimize  # Flushes stale DNS, cleans harmless caches, frees memory
```
