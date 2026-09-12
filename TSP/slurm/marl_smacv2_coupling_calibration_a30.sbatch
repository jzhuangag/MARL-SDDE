#!/bin/bash
#SBATCH --job-name=tsp-smac-cal
#SBATCH --account=vincentlau
#SBATCH --partition=gpu-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a30:1
#SBATCH --time=01:00:00
#SBATCH --array=0-3%4

set -euo pipefail

ROOT=/scratch/jzhuangag/MARL-SDDE-TSP-SMACV2-DEV-001
BASE=/scratch/jzhuangag/MARL-SDDE-TwoClocks-20260902
PYTHON="$BASE/envs/runtime-py39/bin/python"
HARL="$BASE/external/HARL"
CODE="$ROOT/code/MARL-SDDE"
export SC2PATH="$BASE/external/StarCraftII"
export PYTHONPATH="$CODE/TSP/experiments:$BASE/external/SMACv2:$HARL"
export TMPDIR="$ROOT/tmp/coupling-cal-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export MPLCONFIGDIR="$ROOT/tmp/mpl-coupling-cal-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$ROOT/artifacts/coupling-calibration" "$ROOT/logs"

if (( SLURM_ARRAY_TASK_ID < 2 )); then
  COUPLING=independent
  OFFSET=$SLURM_ARRAY_TASK_ID
else
  COUPLING=shared
  OFFSET=$((SLURM_ARRAY_TASK_ID-2))
fi
SEED=$((80501+OFFSET))
OUT="$ROOT/artifacts/coupling-calibration/job${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}-${COUPLING}-seed${SEED}"

"$PYTHON" "$CODE/TSP/experiments/run_smacv2_coupling_calibration.py" \
  --harl-root "$HARL" \
  --output "$OUT" \
  --coupling "$COUPLING" \
  --seed "$SEED" \
  --seed-registry-base 80501 \
  --seed-registry-size 128 \
  --workers 8 \
  --blocks 64 \
  --rollout-length 200 \
  --map-name terran_10_vs_10 \
  --candidate-q 1 8 \
  --correlation-delta 0.05 \
  --server-overhead 800 \
  --cuda

find "$OUT" -type f ! -name SHA256SUMS -print0 | sort -z | \
  xargs -0 sha256sum > "$OUT/SHA256SUMS"
