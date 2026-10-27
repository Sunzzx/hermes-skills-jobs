---
name: kaggle-autopilot
description: "Elite autonomous Kaggle competition intelligence system: research, enter, inspect, engineer features, train GBDTs/ensembles, cross-validate, submit, and climb leaderboards autonomously."
version: 1.0.0
author: Antigravity & Hermes
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [MachineLearning, Kaggle, Autopilot, Competition, LightGBM, XGBoost, Ensembling]
    related_skills: []
---

# Hermes Kaggle Autopilot Skill

## Mission
Autonomous end-to-end competition solver and research engine. Executes the complete lifecycle:
`DISCOVER → EVALUATE → REGISTER → INSPECT → VALIDATE → BASELINE → RESEARCH → EXPERIMENT → OPTIMIZE → ENSEMBLE → AUDIT → SUBMIT → MEASURE → ITERATE → REPORT → LEARN`

## CLI Architecture (`kaggle-autopilot`)
All commands are accessible directly via `kaggle-autopilot` (symlinked in `~/.local/bin/kaggle-autopilot`):

### 1. Scout Competitions
Scans active competitions, filters by ROI, category, and eligibility, and stores opportunities into `~/.hermes/kaggle/ledger.db`:
```bash
kaggle-autopilot scout
```

### 2. Inspect Dataset & Leakage Analysis
Downloads the competition dataset to `~/.hermes/kaggle/workspaces/<comp>/data`, extracts files, inspects train/test distributions, detects target column, ID column, numeric/categorical features, and missing value rates:
```bash
kaggle-autopilot inspect <competition_ref>
```

### 3. Model Training & 5-Fold Cross Validation
Trains adaptive LightGBM / XGBoost baseline models with leak-free out-of-fold (OOF) validation. Automatically selects StratifiedKFold for classification or KFold for regression, evaluates CV metric (AUC, RMSE, etc.), generates submission files, and records scores to the experiment ledger:
```bash
kaggle-autopilot train <competition_ref>
```

### 4. Auditing & Submitting
Audits submission column names, row counts, and nulls against `sample_submission.csv`. If verification passes, submits to Kaggle via the official API:
```bash
kaggle-autopilot submit <competition_ref> --file <path_to_submission.csv> --message "<experiment_note>"
```

### 5. Status Dashboard
Displays entered competitions, running/completed experiments, best CV scores, and leaderboard tracking:
```bash
kaggle-autopilot status
```

## Environment & Storage
* **Python ML Virtualenv**: `~/.hermes/kaggle/.venv` (pre-installed with LightGBM, XGBoost, Scikit-Learn, Scipy, Pandas, NumPy, Kaggle).
* **Workspace Data**: `~/.hermes/kaggle/workspaces/<competition_slug>/`
* **Experiment & Submission Ledger**: SQLite database at `~/.hermes/kaggle/ledger.db`.
* **API Credentials**: Stored securely in `~/.kaggle/access_token` and `~/.hermes/.env` (`KAGGLE_API_TOKEN`).
