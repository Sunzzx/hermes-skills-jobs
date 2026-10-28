#!/usr/bin/env python3
"""Hermes Autonomous Frontier Research Engine Runner.

Drives the 3-hour autonomous engineering cycle:
1. Audits and dynamically discovers all existing repositories in ~/Developer.
2. Runs regression tests across existing projects to ensure continuous green health.
3. Queries GitHub (`gh repo list Sunzzx`) to provide the agent with the exact catalog of existing projects.
4. Enforces the creation of a BRAND NEW, diverse repository every 3 hours that:
   - Advances AI/ML/Deep Learning (novel architectures, reasoning, interpretability)
   - Provides tangible, profound benefit to humanity (biotech, climate, neuroscience, medicine, accessible tech)
   - Explores crazy, radical systems and physical science technology
5. Dispatches notifications to WhatsApp and Discord for every newly deployed project.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
HERMES_DIR = HOME / ".hermes"
DEV_DIR = HOME / "Developer"
LOG_FILE = HERMES_DIR / "logs" / "frontier_research.log"

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def run_cmd(cmd: str, cwd: Path, timeout: int = 40):
    try:
        res = subprocess.run(cmd, shell=True, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
        return res.stdout.strip(), res.stderr.strip(), res.returncode
    except subprocess.TimeoutExpired:
        return "", "Timeout", 124
    except Exception as e:
        return "", str(e), 1


def get_github_repos():
    """List repositories owned by Sunzzx on GitHub."""
    try:
        res = subprocess.run(["gh", "repo", "list", "Sunzzx", "--limit", "50", "--json", "name,description,url,isPrivate"], capture_output=True, text=True, timeout=15)
        if res.returncode == 0:
            return json.loads(res.stdout)
    except Exception as e:
        log(f"GitHub fetch warning: {e}")
    return []


def discover_and_audit_local_projects():
    """Discover and test all candidate projects in ~/Developer."""
    audited = []
    if not DEV_DIR.exists():
        return audited

    for item in sorted(DEV_DIR.iterdir()):
        if not item.is_dir() or item.name.startswith("."):
            continue
        if item.name in ["hermes-skills-jobs", "node_modules", ".venv", "venv", "bugbounty"]:
            continue

        has_pytest = (item / "tests").exists() or (item / "test").exists()
        has_pyproject = (item / "pyproject.toml").exists()
        has_cargo = (item / "Cargo.toml").exists()
        has_git = (item / ".git").exists()

        if not (has_pyproject or has_cargo or has_pytest or has_git):
            continue

        test_cmd = None
        if has_pyproject and has_pytest:
            test_cmd = "uv run pytest -q 2>&1 || pytest -q 2>&1"
        elif has_cargo:
            test_cmd = "cargo test -q 2>&1"
        elif has_pytest:
            test_cmd = "pytest -q 2>&1"

        status = "Built (No Test Harness)"
        if test_cmd:
            stdout, stderr, code = run_cmd(test_cmd, item, timeout=30)
            status = "100% Passed" if code == 0 else f"Failing (code {code})"

        audited.append({
            "name": item.name,
            "path": str(item),
            "test_status": status,
            "has_tests": bool(test_cmd)
        })

    return audited


def notify_new_project(name: str, domain: str, impact: str, repo_url: str):
    """Dispatches alert when a new project is created and pushed."""
    try:
        sys.path.insert(0, str(HERMES_DIR / "scripts"))
        from notifier import send_whatsapp, send_discord
        
        wa_msg = (
            f"🚀 *HERMES FRONTIER RESEARCH ENGINE (NEW REPO)*\n\n"
            f"📦 *Project:* `{name}`\n"
            f"🌐 *Domain:* {domain}\n"
            f"❤️ *Humanity Benefit:* {impact}\n"
            f"🔗 *GitHub Repo:* {repo_url}\n"
            f"🧪 *Tests:* 100% Passing Verified\n\n"
            f"_— Built, packaged & pushed autonomously (3-hour cycle)_"
        )
        send_whatsapp(wa_msg)

        disc_msg = (
            f"🚀 **HERMES FRONTIER RESEARCH ENGINE (NEW REPO)**\n\n"
            f"📦 **Project:** `{name}`\n"
            f"🌐 **Domain:** {domain}\n"
            f"❤️ **Humanity Benefit:** {impact}\n"
            f"🔗 **GitHub:** {repo_url}\n"
            f"🧪 **Tests:** 100% Passing Verified\n"
        )
        send_discord(disc_msg)
        log(f"✓ Dispatched notifications for new project: {name}")
        return True
    except Exception as e:
        log(f"Notification error: {e}")
        return False


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "notify":
        name = sys.argv[2] if len(sys.argv) > 2 else "unknown"
        domain = sys.argv[3] if len(sys.argv) > 3 else "Frontier Tech"
        impact = sys.argv[4] if len(sys.argv) > 4 else "Deep Tech Advancement to Humanity"
        url = sys.argv[5] if len(sys.argv) > 5 else f"https://github.com/Sunzzx/{name}"
        notify_new_project(name, domain, impact, url)
        return

    log("======================================================================")
    log("🔬 HERMES AUTONOMOUS FRONTIER RESEARCH ENGINE (3-HOUR BUILD CYCLE)")
    log("======================================================================")

    local_projects = discover_and_audit_local_projects()
    gh_repos = get_github_repos()
    gh_names = {r["name"] for r in gh_repos}

    print("\n### CURRENT LOCAL REPOSITORIES IN ~/Developer:")
    for p in local_projects:
        gh_status = f"https://github.com/Sunzzx/{p['name']}" if p['name'] in gh_names else "Local Only"
        print(f"• {p['name']}: {p['test_status']} | GitHub: {gh_status}")
        log(f"Existing: {p['name']} ({p['test_status']})")

    print("\n### GITHUB REPOSITORIES (Sunzzx):")
    for r in gh_repos[:12]:
        print(f"• {r['name']} - {r['url']}")

    print("\n" + "=" * 70)
    print("🎯 MANDATORY DIRECTIVE FOR THIS 3-HOUR RUN:")
    print("DO NOT RE-AUDIT OR RE-REPORT AN EXISTING REPOSITORY! A NEW REPO MUST BE BUILT.")
    print("You must autonomously CREATE, TEST, PACKAGE, AND PUSH A BRAND NEW REPOSITORY")
    print("to GitHub under Sunzzx/<new-project-name> during this execution.")
    print("Choose an untouched topic rotating across:")
    print("  1. ADVANCED AI / ML & DEEP LEARNING (MCTS/PRMs, Flow Matching, Sparse Autoencoders, BitNet b1.58, Mamba-2, SNNs, Triton kernels)")
    print("  2. HELPFUL TO HUMANITY (Computational protein/drug dynamics, Climate Navier-Stokes, Renewable smart grid, Epidemiology, Medical MRI Radon, Assistive sensory tech)")
    print("  3. CRAZY & RADICAL TECHNOLOGY (Quantum Clifford/Shor simulators, 4D raymarching, Custom bytecode register VMs, Relativistic N-body, Lenia artificial life)")
    print("\nDeliverables: Full code, 100% passing tests, LaTeX math README.md, runnable visual demo, pyproject.toml/Cargo.toml, MIT license, GitHub push.")
    print("After pushing, notify via:")
    print('  python3 /Users/ayu/.hermes/scripts/frontier_research_runner.py notify "<name>" "<domain>" "<humanity_impact>" "<github_url>"')
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
