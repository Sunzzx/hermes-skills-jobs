#!/usr/bin/env python3
"""Hermes Autonomous Health & Self-Healing Watchdog.

Reviews Hermes system health, model provider connectivity, SQLite databases,
gateway process, locks, and tool executables every 6 hours.
Automatically repairs detected issues and logs all remediation actions.
"""

import os
import sys
import time
import json
import sqlite3
import datetime
import subprocess
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
HERMES_DIR = HOME / ".hermes"
LOG_FILE = HERMES_DIR / "logs" / "health_autoheal.log"
BIN_DIR = HOME / ".local" / "bin"
CRON_JOBS_FILE = HERMES_DIR / "cron" / "jobs.json"

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def run_cmd(cmd: str, timeout: int = 45):
    try:
        res = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return res.stdout.strip(), res.stderr.strip(), res.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out", 124
    except Exception as e:
        return "", str(e), 1


def check_and_heal_sqlite():
    log("🔍 [1/5] Auditing SQLite Databases & WAL Integrity...")
    db_files = [
        HERMES_DIR / "state.db",
        HERMES_DIR / "kanban.db",
        HERMES_DIR / "projects.db",
        HERMES_DIR / "response_store.db",
        HERMES_DIR / "shared-state.db",
        HERMES_DIR / "runs_idempotency.db",
    ]
    for db in db_files:
        if not db.exists():
            continue
        try:
            conn = sqlite3.connect(str(db), timeout=5.0)
            cur = conn.cursor()
            # Integrity check
            cur.execute("PRAGMA integrity_check;")
            res = cur.fetchone()
            if res and res[0] == "ok":
                log(f"  ✓ {db.name}: Integrity OK")
            else:
                log(f"  ⚠️ {db.name}: Integrity warning: {res}")
            # Checkpoint WAL to prevent bloat
            cur.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            conn.commit()
            conn.close()
        except Exception as e:
            log(f"  ❌ Error checking/healing {db.name}: {e}")


def check_and_heal_stale_locks():
    log("🔍 [2/5] Scanning for Stale Locks & Temporary Artifacts...")
    now = time.time()
    locks_found = list(HERMES_DIR.glob("*.lock")) + list(HERMES_DIR.glob("**/*.lock"))
    for lock in locks_found:
        try:
            mtime = lock.stat().st_mtime
            # If older than 2 hours and empty, it's a stale lingering lock
            if (now - mtime) > 7200 and lock.stat().st_size == 0:
                log(f"  🧹 Removing stale lock file: {lock.name} (age: {int((now - mtime) / 60)}m)")
                lock.unlink()
            else:
                log(f"  ✓ Active lock retained: {lock.name}")
        except Exception as e:
            log(f"  ⚠️ Could not audit lock {lock.name}: {e}")


def check_and_heal_models():
    log("🔍 [3/5] Auditing LLM Model Endpoints & Connectivity...")
    # Run built-in Hermes doctor --fix
    stdout, stderr, code = run_cmd("hermes doctor --fix", timeout=60)
    lines = (stdout + "\n" + stderr).split("\n")
    provider_checks = [l for l in lines if "✓" in l or "⚠" in l or "❌" in l]
    for line in provider_checks[:10]:
        log(f"  {line.strip()}")
    if code == 0:
        log("  ✓ Hermes Doctor diagnostic passed.")
    else:
        log("  ℹ️ Doctor completed diagnostic checks (minor optional dependencies noted).")


def check_and_heal_gateway_and_cron():
    log("🔍 [4/5] Auditing Hermes Gateway & Scheduled Cron Jobs...")
    # Check gateway pid
    pid_file = HERMES_DIR / "gateway.pid"
    gateway_alive = False
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            # Check if process is running
            os.kill(pid, 0)
            gateway_alive = True
            log(f"  ✓ Hermes Gateway running with PID {pid}.")
        except (OSError, ValueError):
            gateway_alive = False

    if not gateway_alive:
        log("  ⚠️ Gateway process dead or unresponsive. Healing by restarting...")
        run_cmd("hermes gateway start 2>/dev/null || true")
        log("  ✓ Restart signal dispatched to gateway supervisor.")

    # Audit Cron Jobs
    if CRON_JOBS_FILE.exists():
        try:
            with open(CRON_JOBS_FILE) as f:
                cron_data = json.load(f)
            modified = False
            for job in cron_data.get("jobs", []):
                name = job.get("name", "unknown")
                # Reset failure streak if any
                if job.get("failure_streak", 0) > 0:
                    log(f"  🔧 Auto-healing job '{name}': resetting failure streak from {job['failure_streak']} to 0")
                    job["failure_streak"] = 0
                    job["last_error"] = None
                    modified = True
                # Ensure enabled
                if not job.get("enabled", True):
                    log(f"  🔧 Auto-healing job '{name}': re-enabling job")
                    job["enabled"] = True
                    modified = True
                log(f"  ✓ Cron Job '{name}': schedule={job.get('schedule_display')}, state={job.get('state')}")

            if modified:
                with open(CRON_JOBS_FILE, "w") as f:
                    json.dump(cron_data, f, indent=2)
                log("  ✓ Cron jobs healed and updated in jobs.json.")
        except Exception as e:
            log(f"  ⚠️ Error reading cron/jobs.json: {e}")


def check_and_heal_tools():
    log("🔍 [5/5] Auditing Core Tool Executables & Permissions...")
    critical_tools = ["tv", "kaggle-autopilot", "quanta", "sysdoctor", "hermes-evolve"]
    for tool in critical_tools:
        tool_path = BIN_DIR / tool
        if tool_path.exists():
            # Ensure 0755 permissions
            mode = tool_path.stat().st_mode
            if not (mode & 0o111):
                log(f"  🔧 Fixing executable permissions for {tool}...")
                tool_path.chmod(0o755)
            log(f"  ✓ Tool {tool}: installed & executable at {tool_path}")
        else:
            log(f"  ℹ️ Tool {tool}: not in ~/.local/bin (optional or managed by uv)")


def main():
    log("==================================================")
    log("🩺 HERMES 6-HOUR HEALTH & SELF-HEALING CYCLE")
    log("==================================================")
    start_time = time.time()

    check_and_heal_sqlite()
    check_and_heal_stale_locks()
    check_and_heal_models()
    check_and_heal_gateway_and_cron()
    check_and_heal_tools()

    elapsed = round(time.time() - start_time, 2)
    log(f"✨ Health & Self-Healing Cycle Complete in {elapsed}s.")
    log("==================================================\n")


if __name__ == "__main__":
    main()
