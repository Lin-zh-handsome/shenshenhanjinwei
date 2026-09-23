#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=/home/hanjinwei/miniconda3/envs/math/bin/python
CONFIG=baseline_lnln/config.yaml
if [[ -e outputs/q2/lnln_baseline/best_valid_mae.pt ]]; then
  echo 'Existing LNLN checkpoint found. Preserve or move it before a fresh run.' >&2
  exit 1
fi
"$PYTHON" baseline_lnln/check_contract.py
"$PYTHON" baseline_lnln/train_baseline.py --config "$CONFIG"
"$PYTHON" baseline_lnln/evaluate_baseline.py --config "$CONFIG" --split valid
"$PYTHON" baseline_lnln/evaluate_baseline.py --config "$CONFIG" --split test
