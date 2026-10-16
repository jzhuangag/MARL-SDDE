#!/bin/bash
#SBATCH --job-name=tsp-mam-g0
#SBATCH --account=vincentlau
#SBATCH --partition=gpu-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:a30:1
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2

set -euo pipefail

ROOT=/scratch/jzhuangag/MARL-SDDE-TSP-MAMUJOCO-DEV-001
CODE="$ROOT/code/MARL-SDDE"
BASE=/scratch/jzhuangag/MARL-SDDE-TwoClocks-20260902
PYTHON="$BASE/envs/runtime-py39/bin/python"
HARL="$BASE/external/HARL"
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then COUPLING=independent; SEED=127001; else COUPLING=shared; SEED=127002; fi

export MUJOCO_PY_MUJOCO_PATH="$BASE/external/mujoco210"
export MUJOCO_PY_MJKEY_PATH="$BASE/external/mjkey.txt"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:/usr/lib/nvidia:/usr/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$CODE/TSP/experiments:$HARL"
export PYTHONNOUSERSITE=1
export TMPDIR="$ROOT/tmp/g0-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export MPLCONFIGDIR="$ROOT/tmp/mpl-g0-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$ROOT/g0/$COUPLING" "$ROOT/logs"

"$PYTHON" "$CODE/TSP/experiments/run_marl_mamujoco_controller_g0.py" \
  --harl-root "$HARL" \
  --results-root "$ROOT/g0/$COUPLING/run" \
  --coupling "$COUPLING" \
  --seed "$SEED" \
  --output "$ROOT/g0/$COUPLING/g0.json"
