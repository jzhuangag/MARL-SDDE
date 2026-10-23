#!/bin/bash
#SBATCH --job-name=tspv3-mam-cal
#SBATCH --account=vincentlau
#SBATCH --partition=gpu-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:a30:1
#SBATCH --time=12:00:00
#SBATCH --array=0-7%2

set -euo pipefail

ROOT=/scratch/jzhuangag/MARL-SDDE-TSP-V3-MAMUJOCO-CAL-DEV-001
CODE="$ROOT/code/MARL-SDDE"
BASE=/scratch/jzhuangag/MARL-SDDE-TwoClocks-20260902
PYTHON="$BASE/envs/runtime-py39/bin/python"
HARL="$BASE/external/HARL"
GLOBAL_INDEX=${SLURM_ARRAY_TASK_ID}
if [ "$GLOBAL_INDEX" -lt 0 ] || [ "$GLOBAL_INDEX" -ge 8 ]; then
  echo "GLOBAL_INDEX out of frozen range: $GLOBAL_INDEX" >&2
  exit 2
fi

COUPLING_INDEX=$((GLOBAL_INDEX / 4))
SEED_INDEX=$((GLOBAL_INDEX % 4))
if [ "$COUPLING_INDEX" -eq 0 ]; then COUPLING=independent; else COUPLING=shared; fi
SEED=$((125001+SEED_INDEX))
PROBE_SEED=$((136001+SEED_INDEX))

export MUJOCO_PY_MUJOCO_PATH="$BASE/external/mujoco210"
export MUJOCO_PY_MJKEY_PATH="$BASE/external/mjkey.txt"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:/usr/lib/nvidia:/usr/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$CODE/TSP/experiments:$HARL"
export PYTHONNOUSERSITE=1
export TMPDIR="$ROOT/tmp/${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export MPLCONFIGDIR="$ROOT/tmp/mpl-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$ROOT/artifacts" "$ROOT/logs"

OUT="$ROOT/artifacts/controller-${COUPLING}-seed${SEED}"
"$PYTHON" "$CODE/TSP/experiments/run_mappo_probe_commit.py" \
  --results-root "$OUT" \
  --harl-root "$HARL" \
  --experiment-id TSP-V3-MARL-MAMUJOCO-CAL-DEV-001 \
  --env-name mamujoco \
  --scenario HalfCheetah-v2 \
  --agent-conf 2x3 \
  --continuous-actions \
  --share-param \
  --coupling "$COUPLING" \
  --rollout-length 200 \
  --server-overhead 800 \
  --message-budget 5000000 \
  --environment-budget 1000000 \
  --hidden-size 128 \
  --activation tanh \
  --actor-lr 0.0005 \
  --critic-lr 0.0005 \
  --ppo-epoch 5 \
  --critic-epoch 5 \
  --eval-interval 200 \
  --log-interval 200 \
  --eval-episodes 20 \
  --eval-threads 4 \
  --torch-threads 8 \
  --seed "$SEED" \
  --probe-seed "$PROBE_SEED" \
  --seed-registry-base 125001 \
  --seed-registry-size 16384 \
  --probe-q 8 \
  --probe-blocks 16 \
  --candidate-q 2 8 \
  --correlation-delta 0.05 \
  --cuda

find "$OUT" -type f ! -name SHA256SUMS -print0 | sort -z | \
  xargs -0 sha256sum > "$OUT/SHA256SUMS"

