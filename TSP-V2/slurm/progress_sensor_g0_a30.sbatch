#!/bin/bash
#SBATCH --job-name=tsp-v2-g0
#SBATCH --account=vincentlau
#SBATCH --partition=gpu-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:a30:1
#SBATCH --time=02:00:00
#SBATCH --array=0-7%2

set -euo pipefail

ROOT=/scratch/jzhuangag/MARL-SDDE-TSP-V2-PROGRESS-G0
CODE="$ROOT/code/MARL-SDDE"
BASE=/scratch/jzhuangag/MARL-SDDE-TwoClocks-20260902
PYTHON="$BASE/envs/runtime-py39/bin/python"
HARL="$BASE/external/HARL"
INDEX="$SLURM_ARRAY_TASK_ID"

TASK_INDEX=$((INDEX / 4))
WITHIN=$((INDEX % 4))
COUPLING_INDEX=$((WITHIN / 2))
SEED_INDEX=$((WITHIN % 2))
if [ "$COUPLING_INDEX" -eq 0 ]; then COUPLING=independent; else COUPLING=shared; fi
SEED=$((140101+SEED_INDEX))

export MUJOCO_PY_MUJOCO_PATH="$BASE/external/mujoco210"
export MUJOCO_PY_MJKEY_PATH="$BASE/external/mjkey.txt"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:/usr/lib/nvidia:/usr/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$CODE/TSP-V2/experiments:$CODE/TSP/experiments:$HARL"
export PYTHONNOUSERSITE=1
export TMPDIR="$ROOT/tmp/${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export MPLCONFIGDIR="$ROOT/tmp/mpl-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$ROOT/artifacts" "$ROOT/logs"

COMMON=(
  --harl-root "$HARL"
  --experiment-id TSP-V2-MARL-PROGRESS-G0
  --coupling "$COUPLING"
  --seed "$SEED"
  --candidate-q 1 2 4 8
  --micro-updates 24
  --hidden-size 128
  --activation tanh
  --actor-lr 0.0005
  --critic-lr 0.0005
  --ppo-epoch 5
  --critic-epoch 5
  --torch-threads 8
  --seed-registry-base 140000
  --seed-registry-size 4096
  --share-param
  --cuda
)

if [ "$TASK_INDEX" -eq 0 ]; then
  NAME="mpe-${COUPLING}-seed${SEED}"
  "$PYTHON" "$CODE/TSP-V2/experiments/run_marl_progress_sensor_g0.py" \
    --results-root "$ROOT/artifacts/$NAME" \
    --env-name pettingzoo_mpe --scenario simple_spread_v2 \
    --rollout-length 25 --server-overhead 100 "${COMMON[@]}"
else
  NAME="mamujoco-${COUPLING}-seed${SEED}"
  "$PYTHON" "$CODE/TSP-V2/experiments/run_marl_progress_sensor_g0.py" \
    --results-root "$ROOT/artifacts/$NAME" \
    --env-name mamujoco --scenario HalfCheetah-v2 --agent-conf 2x3 \
    --continuous-actions --rollout-length 200 --server-overhead 800 "${COMMON[@]}"
fi

find "$ROOT/artifacts/$NAME" -type f ! -name SHA256SUMS -print0 | sort -z | \
  xargs -0 sha256sum > "$ROOT/artifacts/$NAME/SHA256SUMS"

