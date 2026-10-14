#!/usr/bin/env bash
set -euo pipefail

# Resumable continuation for the corrected v2 native study.  This script is
# intentionally a shell supervisor: each Python launcher owns its queue and
# writes immutable run records, while this process only advances stages after
# the preceding launcher has exited.

cd "$(dirname "$0")/.."
PY="${PYTHON_BIN:-.venv-study/bin/python}"
HPO="/var/tmp/dora-adaptive-rank/runs/hpo_v2_seed1001"
TUNE="/var/tmp/dora-adaptive-rank/runs/tune_v2_seed1001"
SCREEN="/var/tmp/dora-adaptive-rank/runs/screen_v2_seed1001"
PIPE_HPO="/var/tmp/dora-adaptive-rank/runs/pipeline_hpo_v2_seed1001"
PIPE_TUNE="/var/tmp/dora-adaptive-rank/runs/pipeline_tune_v2_seed1001"

mkdir -p /var/tmp/dora-adaptive-rank/continuation
exec > >(tee -a /var/tmp/dora-adaptive-rank/continuation/native_and_pipeline.log) 2>&1

echo "[$(date -u +%FT%TZ)] continuation started"

# Do not start another native HPO supervisor if the requested one is already
# running.  The loop is also safe when this script is started after HPO has
# completed: it proceeds immediately.
while pgrep -f "scripts/run_hpo_v2.py --root ${HPO}" >/dev/null; do
  echo "[$(date -u +%FT%TZ)] corrected native HPO still running"
  sleep 60
done

echo "[$(date -u +%FT%TZ)] reporting corrected native HPO"
"$PY" scripts/report_v2.py --roots "$HPO" \
  --out reports/hpo_v2.csv --summary reports/hpo_v2_summary.json
"$PY" scripts/plot_v2_progress.py --csv reports/hpo_v2.csv \
  --out reports/figures/hpo_v2.png

echo "[$(date -u +%FT%TZ)] launching/resuming corrected native tuning"
"$PY" scripts/run_tune_v2.py \
  --hpo-root "$HPO" --out-root "$TUNE" \
  --gpus 0,1,2,3 --updates 512 --seeds 1002,1003 \
  --python "$PY" --microbatch 2 --resume
"$PY" scripts/report_v2.py --roots "$TUNE" \
  --out reports/tuning_v2.csv --summary reports/tuning_v2_summary.json
"$PY" scripts/plot_v2_progress.py --csv reports/tuning_v2.csv \
  --out reports/figures/tuning_v2.png

echo "[$(date -u +%FT%TZ)] launching/resuming corrected native screening"
"$PY" scripts/run_screen_v2.py \
  --tune-root "$TUNE" --out-root "$SCREEN" \
  --gpus 0,1,2,3 --updates 512 --seeds 2001,2002,2003 \
  --python "$PY" --microbatch 2 --resume
"$PY" scripts/report_v2.py --roots "$SCREEN" \
  --out reports/screening_v2.csv --summary reports/screening_v2_summary.json
"$PY" scripts/plot_v2_progress.py --csv reports/screening_v2.csv \
  --out reports/figures/screening_v2.png

echo "[$(date -u +%FT%TZ)] launching/resuming pipeline HPO"
"$PY" scripts/run_hpo_pipeline_v2.py \
  --root "$PIPE_HPO" --data-root /var/tmp/dora-adaptive-rank/data_v2 \
  --manifest-root artifacts/manifests_v2 --updates 128 --seed 1001 \
  --gpus 0,1,2,3 --python "$PY" --microbatch 2 --resume
"$PY" scripts/report_v2.py --roots "$PIPE_HPO" \
  --out reports/pipeline_hpo_v2.csv --summary reports/pipeline_hpo_v2_summary.json

echo "[$(date -u +%FT%TZ)] launching/resuming pipeline tuning"
"$PY" scripts/run_tune_pipeline_v2.py \
  --hpo-root "$PIPE_HPO" --out-root "$PIPE_TUNE" \
  --gpus 0,1,2,3 --updates 512 --seeds 1002,1003 \
  --python "$PY" --microbatch 2 --resume
"$PY" scripts/report_v2.py --roots "$PIPE_TUNE" \
  --out reports/pipeline_tuning_v2.csv --summary reports/pipeline_tuning_v2_summary.json

echo "[$(date -u +%FT%TZ)] continuation reached pipeline tuning terminal state"
