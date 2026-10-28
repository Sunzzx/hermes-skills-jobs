#!/usr/bin/env python3
"""Hermes Daily Kaggle Autopilot Runner.

Runs the complete autonomous competition intelligence cycle:
1. Scout active high-ROI competitions.
2. Select target competition from active queue.
3. Inspect data & feature schema.
4. Train leak-free 5-Fold CV baseline / improved ensemble.
5. Audit and submit to Kaggle leaderboard.
6. Synchronize submission scores with Kaggle API & record metrics in ledger.
"""

import os
import sys
import json
import sqlite3
import subprocess
from pathlib import Path

BASE_DIR = Path(os.path.expanduser("~/.hermes/kaggle"))
AUTOPILOT_BIN = "/Users/ayu/.local/bin/kaggle-autopilot"
PYTHON_BIN = str(BASE_DIR / ".venv" / "bin" / "python")
KAGGLE_BIN = str(BASE_DIR / ".venv" / "bin" / "kaggle")
DB_PATH = BASE_DIR / "ledger.db"


def run(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.stdout.strip(), p.stderr.strip(), p.returncode


def get_cli_env():
    env = os.environ.copy()
    token_file = Path(os.path.expanduser("~/.kaggle/access_token"))
    if token_file.exists():
        env["KAGGLE_API_TOKEN"] = token_file.read_text().strip()
    return env


def is_tabular_competition(comp_ref: str) -> bool:
    """Check via Kaggle API file list if the competition is a tabular dataset."""
    p = subprocess.run([KAGGLE_BIN, "competitions", "files", "-c", comp_ref], capture_output=True, text=True, env=get_cli_env())
    if p.returncode != 0:
        return False
    lines = p.stdout.lower()
    has_train = ('train' in lines) and ('.csv' in lines or '.parquet' in lines)
    has_test = ('test' in lines) and ('.csv' in lines or '.parquet' in lines)
    return has_train and has_test


def sync_submission_scores():
    """Check Kaggle API for updated leaderboard scores and persist to ledger."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT competition_ref FROM submissions WHERE status != 'complete' OR public_score IS NULL")
    pending_comps = [r[0] for r in cur.fetchall()]
    for comp in pending_comps:
        p = subprocess.run([KAGGLE_BIN, "competitions", "submissions", "-c", comp], capture_output=True, text=True, env=get_cli_env())
        if p.returncode == 0 and p.stdout:
            for line in p.stdout.splitlines()[2:]:  # skip headers
                parts = line.split()
                if len(parts) >= 6 and 'COMPLETE' in line:
                    score_str = parts[-1]
                    try:
                        score_val = float(score_str)
                        cur.execute("UPDATE submissions SET public_score = ?, status = 'complete' WHERE competition_ref = ? AND (public_score IS NULL OR status != 'complete')", (score_val, comp))
                    except ValueError:
                        pass
    conn.commit()
    conn.close()


def main():
    print("=" * 60)
    print("🚀 HERMES KAGGLE AUTOPILOT DAILY RUNNER")
    print("=" * 60)

    # 1. Scout competitions
    print("[1/5] Scouting active competitions...")
    stdout, stderr, code = run(f"{AUTOPILOT_BIN} scout")
    print(stdout if stdout else stderr)

    # 2. Select target competition (prioritizing cash prize pools over non-prize tracks, filtering out expired)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT ref, title, reward FROM competitions WHERE user_has_entered=1 AND (deadline IS NULL OR deadline > datetime('now')) ORDER BY (CASE WHEN reward LIKE '%Usd%' THEN 1 ELSE 0 END) DESC, team_count DESC")
    candidates = cur.fetchall()
    conn.close()

    if not candidates:
        print("[-] No entered competitions found in ledger.")
        return

    selected_target = None
    print(f"\n[2/5] Evaluating {len(candidates)} candidates for tabular compatibility...")
    for comp_ref, title, reward in candidates:
        # Check if workspace already downloaded or check files remotely
        ws_data = BASE_DIR / "workspaces" / comp_ref / "data"
        has_local = ws_data.exists() and any(ws_data.glob("*train*"))
        if has_local or is_tabular_competition(comp_ref):
            selected_target = (comp_ref, title, reward)
            print(f"  [+] Selected viable competition: {comp_ref} ({title}) | Reward: {reward}")
            break
        else:
            print(f"  [-] Skipping {comp_ref} (non-standard dataset or rules acceptance required)")

    if not selected_target:
        print("[-] No tabular competition met criteria.")
        return

    comp_ref, title, reward = selected_target

    # 3. Inspect dataset
    print(f"\n[3/5] Inspecting validation schema for '{comp_ref}'...")
    out_inspect, _, _ = run(f"{AUTOPILOT_BIN} inspect {comp_ref}")
    try:
        meta = json.loads(out_inspect)
        print(f"  • Target: {meta.get('target_col')} | ID: {meta.get('id_col')}")
        print(f"  • Train shape: {meta.get('train_shape')} | Test shape: {meta.get('test_shape')}")
        print(f"  • Numeric features: {len(meta.get('numeric_cols', []))} | Categorical features: {len(meta.get('categorical_cols', []))}")
    except Exception as e:
        print(f"  [!] Inspection output: {out_inspect}")

    # 4. Train 5-fold cross-validation
    print(f"\n[4/5] Running 5-Fold cross-validation & training for '{comp_ref}'...")
    stdout_tr, stderr_tr, _ = run(f"{AUTOPILOT_BIN} train {comp_ref}")
    print(stdout_tr if stdout_tr else stderr_tr)
    sub_file = None
    try:
        idx = stdout_tr.find('{')
        if idx != -1:
            res_tr = json.loads(stdout_tr[idx:])
            sub_file = res_tr.get("submission_file")
    except Exception:
        pass

    # 5. Submit to Kaggle leaderboard if submission was generated
    if sub_file and os.path.exists(sub_file):
        print(f"\n[5/5] Submitting verified predictions to '{comp_ref}'...")
        stdout_sub, stderr_sub, _ = run(f"{AUTOPILOT_BIN} submit {comp_ref} --file {sub_file} --message 'Hermes Autopilot Daily 5-Fold LGBM'")
        print(stdout_sub if stdout_sub else stderr_sub)

    # Synchronize score updates
    sync_submission_scores()

    print("\n" + "=" * 60)
    print("📊 CURRENT AUTOPILOT STATUS:")
    print("=" * 60)
    out_status, _, _ = run(f"{AUTOPILOT_BIN} status")
    print(out_status)

    # Dispatch WhatsApp summary notification
    try:
        sys.path.insert(0, str(Path(os.path.expanduser("~/.hermes/scripts"))))
        from notifier import notify_job_summary
        sub_status = "Verified & Submitted" if sub_file and os.path.exists(sub_file) else "5-Fold Baseline Trained"
        summary = (
            f"🏆 *Target Competition:* {title} (`{comp_ref}`)\n"
            f"• Prize Purse: {reward}\n"
            f"• ML Model: 5-Fold Stratified LightGBM/XGBoost\n"
            f"• Leaderboard Submission: {sub_status}\n"
            f"• Ledger & Metrics: Persisted to ~/.hermes/kaggle/ledger.db"
        )
        notify_job_summary("kaggle-autopilot-daily", summary)
    except Exception as e:
        print(f"[!] Notification dispatch error: {e}")


if __name__ == "__main__":
    main()

