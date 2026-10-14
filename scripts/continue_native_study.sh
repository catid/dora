#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONPATH=src HF_HOME=/var/tmp/dora-adaptive-rank/hf
HPO=/var/tmp/dora-adaptive-rank/runs/hpo_accepted_final2
TUNE=/var/tmp/dora-adaptive-rank/runs/tune_accepted_final2
SCREEN=/var/tmp/dora-adaptive-rank/runs/screen_accepted_final2
while pgrep -f 'scripts/run_hpo_native.py.*hpo_accepted_final2' >/dev/null; do sleep 60; done
$PY scripts/run_hpo_native.py --gpus 0,1,2,3 --updates 128 --root "$HPO" --resume
$PY scripts/run_selected_native.py --hpo-root "$HPO" --out-root "$TUNE" --gpus 0,1,2,3 --updates 512 --seeds 1002,1003 --python "$PY" --resume
$PY scripts/run_screen_native.py --tune-root "$TUNE" --out-root "$SCREEN" --gpus 0,1,2,3 --updates 512 --seeds 2001,2002,2003 --python "$PY"
$PY scripts/report_native.py --root "$HPO" --out reports/hpo.csv
$PY scripts/report_native.py --root "$TUNE" --out reports/tuning.csv
$PY scripts/report_native.py --root "$SCREEN" --out reports/screening.csv
