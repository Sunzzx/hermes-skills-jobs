import os, sys, json, datetime, re, shutil, subprocess
from pathlib import Path
repo = Path('/Users/ayu/Developer/hermes-skills-jobs')
os.chdir(repo)

# 1. bin/ssd-engine
bin_path = repo / 'bin' / 'ssd-engine'
bin_path.parent.mkdir(parents=True, exist_ok=True)
ssd_script = '''#!/usr/bin/env python3
import sys, argparse, math, numpy as np

def verify():
    rng = np.random.RandomState(42)
    B, T, H, D, P = 2, 32, 4, 16, 16
    X = rng.randn(B, T, H, D)
    Bin = rng.randn(B, T, H, P)
    Cin = rng.randn(B, T, H, P)
    dt = rng.uniform(0.01, 0.05, size=(B, T, H))
    A = -np.exp(rng.uniform(math.log(0.1), math.log(5.0), size=(H,)))
    
    h = np.zeros((B, H, P, D))
    Y_rec = np.zeros((B, T, H, D))
    for t in range(T):
        alpha = np.exp(dt[:, t, :] * A[None, :])
        h = h * alpha[:, :, None, None] + Bin[:, t, :, :, None] * X[:, t, :, None, :]
        Y_rec[:, t, :, :] = np.sum(Cin[:, t, :, :, None] * h, axis=2)
        
    Lambda = np.cumsum(dt * A[None, None, :], axis=1)
    Y_attn = np.zeros((B, T, H, D))
    mask = np.tril(np.ones((T, T), dtype=bool))
    for b in range(B):
        for h_idx in range(H):
            lam = Lambda[b, :, h_idx]
            diff = lam[:, None] - lam[None, :]
            M = np.where(mask, np.exp(diff), 0.0)
            S = M * (Cin[b, :, h_idx, :] @ Bin[b, :, h_idx, :].T)
            Y_attn[b, :, h_idx, :] = S @ X[b, :, h_idx, :]
            
    diff = float(np.max(np.abs(Y_rec - Y_attn)))
    print(f'Max diff recurrent vs dual attention: {diff:.3e}')
    assert diff < 1e-11, f'Mismatch: {diff}'
    print('State Space Duality verification passed!')

def main():
    p0 = sys.argv[1] if len(sys.argv) > 1 else 'verify'
    if p0 == 'verify':
        verify()
    elif p0 == 'inspect':
        print('State Space Duality (SSD) & Mamba-2 SSM-Attention Engine')

if __name__ == '__main__':
    main()
'''
bin_path.write_text(ssd_script.strip() + '\n')
bin_path.chmod(0o755)

local_bin = Path.home() / '.local' / 'bin' / 'ssd-engine'
local_bin.parent.mkdir(parents=True, exist_ok=True)
local_bin.write_text(ssd_script.strip() + '\n')
local_bin.chmod(0o755)

# 2-a. tests/test_ssd.py
test_dir = repo / 'tests'
test_dir.mkdir(parents=True, exist_ok=True)
test_path = test_dir / 'test_ssd.py'
test_script = '''import unittest, subprocess, sys, os
class TestSSD(unittest.TestCase):
    def test_verify_cli(self):
        res = subprocess.run([sys.executable, 'bin/ssd-engine', 'verify'], capture_output=True, text=True, check=True)
        self.assertIn('verification passed', res.stdout)

if __name__ == '__main__':
    unittest.main()
'''
test_path.write_text(test_script.strip() + '\n')
subprocess.run([sys.executable, '-m', 'pytest', str(test_path)], check=True)

# 2-b. skills/systems/state-space-duality/SKILL.md
skill_dir = repo / 'skills' / 'systems' / 'state-space-duality'
skill_dir.mkdir(parents=True, exist_ok=True)
skill_md = '''---
name: state-space-duality
description: "State Space Duality (SSD) & Mamba-2 SSM-Attention Engine: exact mathematical equivalence between continuous/discretized 1D SSMs and causal linear attention with 1-semiseparable decay masks."
category: systems
type: skill+tool
cli: ssd-engine
tags:
  - mamba-2
  - ssm
  - state-space-duality
  - linear-attention
---

# State Space Duality (SSD) & Mamba-2 SSM-Attention Engine

Exact mathematical duality proving SSMs are attention under decay masking.

## Commands
```bash
ssd-engine verify
ssd-engine inspect
```
'''
(skill_dir / 'SKILL.md').write_text(skill_md.strip() + '\n')
local_skill_dir = Path.home() / '.hermes' / 'skills' / 'systems' / 'state-space-duality'
local_skill_dir.mkdir(parents=True, exist_ok=True)
(local_skill_dir / 'SKILL.md').write_text(skill_md.strip() + '\n')

# 3. registry.json
reg_file = repo / 'registry.json'
with open(reg_file, 'r') as f:
    reg = json.load(f)
reg = [r for r in reg if r.get('id') != 'state-space-duality']
reg.append({
    'id': 'state-space-duality',
    'title': 'State Space Duality (SSD) & Mamba-2 SSM-Attention Engine',
    'category': 'systems',
    'type': 'skill+tool',
    'description': 'Exact mathematical duality between structured SSMs and causal linear attention with 1-semiseparable decay masks.',
    'tags': ['mamba-2', 'ssm', 'state-space-duality', 'linear-attention'],
    'cli': 'ssd-engine',
    'evolved_at': datetime.datetime.now().isoformat(),
})
with open(reg_file, 'w') as f:
    json.dump(reg, f, indent=2)

# 4. CHANGELOG.md & README.md
cl_file = repo / 'CHANGELOG.md'
cl = cl_file.read_text()
if 'state-space-duality' not in cl:
    cl_entry = '\n## [2026-11-03 16:30] - State Space Duality (SSD) & Mamba-2 SSM-Attention Engine (`state-space-duality`)\n- **Title**: State Space Duality (SSD) & Mamba-2 SSM-Attention Engine\n- **Category**: systems\n- **Description**: Exact mathematical duality between structured SSMs and causal linear attention with 1-semiseparable decay masks.\n- **CLI**: `ssd-engine` (`verify`, `inspect`)\n'
    cl_file.write_text(cl.rstrip() + cl_entry)

readme_file = repo / 'README.md'
readme = readme_file.read_text()
readme = re.sub(r'Total Skills\*\*: \d+', f'Total Skills**: {len(reg)}', readme)
readme = re.sub(r'\*Last Evolved: [^\*]+\*', f'*Last Evolved: {datetime.datetime.now().strftime("%Y-%m-%d %H:MM:%S")}*', readme)
if 'state-space-duality' not in readme:
    row = '| **State Space Duality (SSD) & Mamba-2 SSM-Attention Engine** | `systems` | `ssd-engine` | Exact mathematical duality between structured SSMs and causal linear attention with 1-semiseparable decay masks | `mamba-2` `ssm` `state-space-duality` `linear-attention` |\n'
    lines = readme.splitlines()
    for i, l in enumerate(lines):
        if l.startswith('| **Multi-Head'):
            lines.insert(i+1, row.strip())
            break
    readme = '\n'.join(lines) + '\n'
readme_file.write_text(readme)

# 5. Git commit & push
subprocess.run(['git', 'add', '-A'], check=True)
subprocess.run(['git', 'commit', '-m', 'feat(evolution): evolve State Space Duality (SSD) & Mamba-2 SSM-Attention Engine [state-space-duality]'], check=True)
subprocess.run(['git', 'push', 'origin', 'main'], check=True)
print('SUCCESS_EVOLVED_AND_PUSHED')
