#!/bin/bash
#SBATCH --job-name=tsp-mtask-g0
#SBATCH --account=vincentlau
#SBATCH --partition=gpu-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a30:1
#SBATCH --time=00:20:00
#SBATCH --array=0-1

set -euo pipefail

ROOT=/scratch/jzhuangag/MARL-SDDE-TSP-MULTITASK-PHASE-A-001
CODE="$ROOT/code/MARL-SDDE"
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  TASK_ID=mpe_speaker_listener
  BASE=/scratch/jzhuangag/MARL-SDDE-TSP-MARL-CONF-001
  PYTHON="$BASE/venv/bin/python"
  HARL="$BASE/tmp/HARL"
else
  TASK_ID=mamujoco_halfcheetah_2x3
  BASE=/scratch/jzhuangag/MARL-SDDE-TwoClocks-20260902
  PYTHON="$BASE/envs/runtime-py39/bin/python"
  HARL="$BASE/external/HARL"
fi
export PYTHONPATH="$CODE/TSP/experiments:$HARL"
export TMPDIR="$ROOT/tmp/${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export MPLCONFIGDIR="$ROOT/tmp/mpl-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$ROOT/g0" "$ROOT/logs"

"$PYTHON" "$CODE/TSP/experiments/audit_multitask_phase_compatibility.py" \
  --harl-root "$HARL" \
  --config "$CODE/TSP/experiments/marl_multitask_phase_headroom.json" \
  --task-id "$TASK_ID" \
  --output "$ROOT/g0/compatibility-${TASK_ID}.json"

sha256sum "$ROOT/g0/compatibility-${TASK_ID}.json" \
  > "$ROOT/g0/SHA256SUMS-${TASK_ID}"
