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

set -euo pipefail

ROOT=/scratch/jzhuangag/MARL-SDDE-TSP-MULTITASK-PHASE-A-001
BASE=/scratch/jzhuangag/MARL-SDDE-TwoClocks-20260902
PYTHON="$BASE/envs/runtime-py39/bin/python"
HARL="$BASE/external/HARL"
CODE="$ROOT/code/MARL-SDDE"
export PYTHONPATH="$CODE/TSP/experiments:$HARL"
export TMPDIR="$ROOT/tmp/${SLURM_JOB_ID}"
export MPLCONFIGDIR="$ROOT/tmp/mpl-${SLURM_JOB_ID}"
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$ROOT/g0" "$ROOT/logs"

"$PYTHON" "$CODE/TSP/experiments/audit_multitask_phase_compatibility.py" \
  --harl-root "$HARL" \
  --config "$CODE/TSP/experiments/marl_multitask_phase_headroom.json" \
  --output "$ROOT/g0/compatibility.json"

sha256sum "$ROOT/g0/compatibility.json" > "$ROOT/g0/SHA256SUMS"
