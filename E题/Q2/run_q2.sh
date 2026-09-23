#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=/home/hanjinwei/miniconda3/envs/math/bin/python
CONFIG=configs/q2_aligned.yaml
"$PYTHON" verify_minimal.py --config "$CONFIG" --stage data
"$PYTHON" train.py --config "$CONFIG"
"$PYTHON" evaluate.py --config "$CONFIG" --checkpoint outputs/q2/best_joint.pt --split valid
"$PYTHON" evaluate.py --config "$CONFIG" --checkpoint outputs/q2/best_joint.pt --split test
"$PYTHON" robustness_sweep.py --config "$CONFIG" --checkpoint outputs/q2/best_joint.pt
"$PYTHON" ablation.py --config "$CONFIG"
"$PYTHON" infer_attachment3.py --config "$CONFIG" --checkpoint outputs/q2/best_joint.pt
"$PYTHON" export_q2_report_tables.py --config "$CONFIG"
"$PYTHON" verify_minimal.py --config "$CONFIG" --stage final
