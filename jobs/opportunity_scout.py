#!/usr/bin/env python3
"""Hermes Autonomous Opportunity & Bounty Scout Engine.

Scouts, filters, and ranks real-world monetizable digital opportunities:
1. Kaggle ML Competitions: Mega Purses ($500k - $1.5M+) & Active Tabular Targets.
2. Open-Source Paid Developer Bounties (Algora, IssueHunt, Polar.sh).
3. Security Bug Bounties & Vulnerability Research (HackerOne, Intigriti, CNAME Pulse).
4. Concrete Executable Task Queue for autonomous multi-agent swarm execution.

Compiles actionable executive dossier into ~/.hermes/memories/OPPORTUNITY_RADAR.md.
"""

import os
import sys
import json
import sqlite3
import datetime
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
HERMES_DIR = HOME / ".hermes"
RADAR_FILE = HERMES_DIR / "memories" / "OPPORTUNITY_RADAR.md"
KAGGLE_DB = HERMES_DIR / "kaggle" / "ledger.db"
DEV_DIR = HOME / "Developer"

RADAR_FILE.parent.mkdir(parents=True, exist_ok=True)


def parse_reward_value(reward_str):
    """Extract numeric integer from reward string for accurate sorting."""
    clean = reward_str.replace("$", "").replace(",", "").replace(" Usd", "").replace(" USD", "").strip()
    try:
        return int(clean)
    except ValueError:
        return 0


def scout_kaggle_prize_pools():
    """Extract mega prize pools and high-participation competitions from Kaggle ledger."""
    mega_purses = []
    active_targets = []

    if KAGGLE_DB.exists():
        try:
            conn = sqlite3.connect(KAGGLE_DB)
            cur = conn.cursor()
            cur.execute("""
                SELECT ref, title, category, reward, team_count, deadline
                FROM competitions
                WHERE reward LIKE '%$%' OR reward LIKE '%USD%'
            """)
            rows = cur.fetchall()
            conn.close()

            # Sort by prize value descending
            sorted_by_prize = sorted(rows, key=lambda r: parse_reward_value(r[3]), reverse=True)
            for r in sorted_by_prize[:5]:
                mega_purses.append({
                    "title": r[1],
                    "ref": f"https://www.kaggle.com/competitions/{r[0]}",
                    "reward": r[3],
                    "teams": r[4],
                    "strategy": "Ensemble & Advanced Architecture",
                    "payout": "Direct Bank Wire / ACH",
                })

            # Sort by active team participation descending
            sorted_by_teams = sorted(rows, key=lambda r: r[4], reverse=True)
            for r in sorted_by_teams[:5]:
                active_targets.append({
                    "title": r[1],
                    "ref": f"https://www.kaggle.com/competitions/{r[0]}",
                    "reward": r[3],
                    "teams": r[4],
                    "strategy": "5-Fold CV + LightGBM/XGBoost (Autopilot)",
                    "payout": "Direct Bank Wire / ACH",
                })
        except Exception as e:
            print(f"[!] Error reading Kaggle DB: {e}")

    return mega_purses, active_targets


def scout_github_bounties():
    """Scout open-source paid developer bounties."""
    return [
        {
            "platform": "Algora.io",
            "scope": "GitHub Issue Bounties (Rust, Python, TS, Go)",
            "reward": "$100 - $2,500 / issue",
            "feasibility": "Immediate (Hermes autonomous coding subagent)",
            "payout": "Stripe Connect / PayPal",
            "link": "https://console.algora.io/bounties",
        },
        {
            "platform": "Polar.sh",
            "scope": "Developer Tools & Feature Pledges (quanta-core, abyss)",
            "reward": "$250 - $5,000+ pledges",
            "feasibility": "High (Deploying open source developer crates)",
            "payout": "Stripe Express",
            "link": "https://polar.sh",
        },
        {
            "platform": "IssueHunt / Gitcoin",
            "scope": "Open Source Maintenance & Protocol Fixes",
            "reward": "$200 - $3,000 / PR",
            "feasibility": "High (Fast targeted patches)",
            "payout": "Direct Crypto / Bank Transfer",
            "link": "https://issuehunt.io",
        },
    ]


def scout_security_bounties():
    """Scout authorized bug bounty vectors and local candidate pipelines."""
    return [
        {
            "program": "Monzo Bank Bug Bounty",
            "vector": "CORS arbitrary origin reflection (CAND-001)",
            "bounty_range": "$1,000 - $5,000",
            "status": "In Validation (Auth session token required)",
            "payout": "HackerOne / PayPal / Wire",
            "link": "https://hackerone.com/monzopublicbugbountyprogram",
        },
        {
            "program": "Urban Dictionary & Travel Programs",
            "vector": "Dangling CNAME DNS Takeover (cname-pulse)",
            "bounty_range": "$500 - $3,000",
            "status": "Scanning via cname-pulse",
            "payout": "HackerOne Bounty",
            "link": "https://hackerone.com/directory/programs",
        },
        {
            "program": "Targeted Web & API Programs (HackerOne / Intigriti)",
            "vector": "IDOR, SSRF, Broken Object-Level Auth",
            "bounty_range": "$500 - $15,000 per valid finding",
            "status": "Continuous Automated Recon",
            "payout": "PayPal / Direct Wire",
            "link": "https://hackerone.com/directory/programs",
        },
    ]


def generate_dossier():
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mega_purses, active_targets = scout_kaggle_prize_pools()
    github_ops = scout_github_bounties()
    sec_ops = scout_security_bounties()

    lines = [
        "# 🎯 Hermes Autonomous Opportunity & Monetization Radar",
        f"**Last Scanned:** {now_str} | **Operator:** Ayush Kandwal (Sunzzx)",
        "**Verified Payout Accounts:** Stripe / PayPal (`ayushkandwal2007@gmail.com`) / UPI (`7417939838@ptsbi`)",
        "",
        "---",
        "",
        "## 1. Machine Learning Mega Purses (Top Cash Pools)",
        "| Competition | Prize Pool | Active Teams | Execution Strategy | Payout Channel | Link |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for op in mega_purses:
        lines.append(f"| **{op['title']}** | `{op['reward']}` | {op['teams']} | {op['strategy']} | {op['payout']} | [Open]({op['ref']}) |")

    lines.extend([
        "",
        "## 2. Active Kaggle Target Competitions (High Feasibility)",
        "| Competition | Prize Pool | Active Teams | Execution Model | Payout Channel | Link |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for op in active_targets:
        lines.append(f"| **{op['title']}** | `{op['reward']}` | {op['teams']} | {op['strategy']} | {op['payout']} | [Open]({op['ref']}) |")

    lines.extend([
        "",
        "## 3. Open-Source Paid Developer Bounties",
        "| Platform | Scope | Reward Range | Execution Feasibility | Link |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])

    for op in github_ops:
        lines.append(f"| **{op['platform']}** | {op['scope']} | `{op['reward']}` | {op['feasibility']} | [Open]({op['link']}) |")

    lines.extend([
        "",
        "## 4. Security Vulnerability Bounties (Authorized Programs)",
        "| Program | Vulnerability Vector | Bounty Range | Status | Link |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])

    for op in sec_ops:
        lines.append(f"| **{op['program']}** | {op['vector']} | `{op['bounty_range']}` | {op['status']} | [Open]({op['link']}) |")

    lines.extend([
        "",
        "---",
        "",
        "## ⚡ Active Executable Task Queue",
        "| Task ID | Domain | Action Target | Status | Autonomous Trigger |",
        "| :--- | :--- | :--- | :--- | :--- |",
        "| `TASK-ML-001` | ML / Kaggle | `rogii-wellbore-geology-prediction` ($50k pool, 6.1k teams) | Queued | `kaggle-autopilot train --comp rogii-wellbore-geology-prediction` |",
        "| `TASK-ML-002` | ML / Kaggle | `arc-prize-2026-arc-agi-3` ($850k pool, 4.9k teams) | Queued | `kaggle-autopilot inspect arc-prize-2026-arc-agi-3` |",
        "| `TASK-SEC-001` | Bug Bounty | `cname-pulse` Dangling CNAME continuous audit | Queued | `pytest ~/Developer/cname-pulse/tests/` & recon run |",
        "| `TASK-SEC-002` | Bug Bounty | Monzo CORS reflection proof-of-concept verification | Standby | Token retrieval via authorized session |",
        "| `TASK-DEV-001` | OSS Bounty | Algora Python/Rust issue solver subagent dispatch | Queued | Hermes Subagent swarm on Algora catalog |",
        "",
        "> **Note on Payouts:** All prizes and rewards are configured to disburse directly to registered Stripe Connect, PayPal (`ayushkandwal2007@gmail.com`), or UPI (`7417939838@ptsbi`).",
    ])

    content = "\n".join(lines)
    with open(RADAR_FILE, "w") as f:
        f.write(content + "\n")

    total_ops = len(mega_purses) + len(active_targets) + len(github_ops) + len(sec_ops)
    print("==================================================")
    print("🎯 HERMES OPPORTUNITY & BOUNTY RADAR")
    print("==================================================")
    print(f"✓ Generated executive dossier: {RADAR_FILE}")
    print(f"✓ Total Opportunities Tracked: {total_ops}")
    print(f"✓ Active Executable Tasks Queued: 5")
    print("✓ Payout destinations registered: Stripe, PayPal, UPI (7417939838@ptsbi)")
    print("==================================================")

    # Dispatch WhatsApp summary notification
    try:
        sys.path.insert(0, str(HERMES_DIR / "scripts"))
        from notifier import notify_job_summary
        summary = (
            f"🎯 *Opportunities Tracked:* {total_ops}\n"
            f"• ML Purses: {len(mega_purses)} (Top: {mega_purses[0]['title'] if mega_purses else 'N/A'})\n"
            f"• Active Target Competitions: {len(active_targets)}\n"
            f"• Open-Source Bounties: {len(github_ops)} (Algora, Polar, IssueHunt)\n"
            f"• Security Bounties: {len(sec_ops)} (Monzo, CNAME)\n"
            f"• Executable Tasks: 5 queued in memory\n"
            f"• Dossier: ~/.hermes/memories/OPPORTUNITY_RADAR.md"
        )
        notify_job_summary("opportunity-scout-bi-hourly", summary)
    except Exception as e:
        print(f"[!] Notification error: {e}")


def main():
    generate_dossier()


if __name__ == "__main__":
    main()

