#!/bin/bash
#SBATCH --job-name=tsp-mtask-a
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

ROOT=/scratch/jzhuangag/MARL-SDDE-TSP-MULTITASK-PHASE-A-001
CODE="$ROOT/code/MARL-SDDE"
GLOBAL_INDEX=$((${OFFSET:?OFFSET must be exported}+SLURM_ARRAY_TASK_ID))
if [ "$GLOBAL_INDEX" -lt 0 ] || [ "$GLOBAL_INDEX" -ge 32 ]; then
  echo "GLOBAL_INDEX out of frozen range: $GLOBAL_INDEX" >&2
  exit 2
fi
TASK_INDEX=$((GLOBAL_INDEX / 16))
LOCAL_INDEX=$((GLOBAL_INDEX % 16))
RAY_INDEX=$((LOCAL_INDEX / 8))
COUPLING_INDEX=$(((LOCAL_INDEX % 8) / 4))
Q_INDEX=$(((LOCAL_INDEX % 4) / 2))
SEED_INDEX=$((LOCAL_INDEX % 2))

if [ "$RAY_INDEX" -eq 0 ]; then
  RAY=message_binding
  MESSAGE_BUDGET=5000000
else
  RAY=environment_binding
  MESSAGE_BUDGET=12000000
fi
if [ "$COUPLING_INDEX" -eq 0 ]; then
  COUPLING=independent
else
  COUPLING=shared
fi
if [ "$Q_INDEX" -eq 0 ]; then
  Q=1
else
  Q=8
fi
SEED=$((112001+SEED_INDEX))

if [ "$TASK_INDEX" -eq 0 ]; then
  TASK_ID=mpe_speaker_listener
  BASE=/scratch/jzhuangag/MARL-SDDE-TSP-MARL-CONF-001
  PYTHON="$BASE/venv/bin/python"
  HARL="$BASE/tmp/HARL"
  TASK_ARGS=(
    --env-name pettingzoo_mpe
    --scenario simple_speaker_listener_v3
    --no-continuous-actions
    --no-share-param
    --rollout-length 25
    --server-overhead 100
    --hidden-size 64
    --activation tanh
    --ppo-epoch 2
    --critic-epoch 2
    --eval-interval 1000
    --log-interval 1000
    --eval-episodes 32
  )
else
  TASK_ID=mamujoco_halfcheetah_2x3
  BASE=/scratch/jzhuangag/MARL-SDDE-TwoClocks-20260902
  PYTHON="$BASE/envs/runtime-py39/bin/python"
  HARL="$BASE/external/HARL"
  export MUJOCO_PY_MUJOCO_PATH="$BASE/external/mujoco210"
  export MUJOCO_PY_MJKEY_PATH="$BASE/external/mjkey.txt"
  export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:/usr/lib/nvidia:/usr/lib64:${LD_LIBRARY_PATH:-}"
  TASK_ARGS=(
    --env-name mamujoco
    --scenario HalfCheetah-v2
    --agent-conf 2x3
    --continuous-actions
    --share-param
    --rollout-length 200
    --server-overhead 800
    --hidden-size 128
    --activation tanh
    --ppo-epoch 5
    --critic-epoch 5
    --eval-interval 200
    --log-interval 200
    --eval-episodes 20
  )
fi

export PYTHONPATH="$CODE/TSP/experiments:$HARL"
export PYTHONNOUSERSITE=1
export TMPDIR="$ROOT/tmp/stage-a-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export MPLCONFIGDIR="$ROOT/tmp/mpl-stage-a-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$ROOT/artifacts/stage-a" "$ROOT/logs"
OUT="$ROOT/artifacts/stage-a/${TASK_ID}-${RAY}-${COUPLING}-q${Q}-seed${SEED}"

"$PYTHON" "$CODE/TSP/experiments/run_mappo_return_bridge.py" \
  --harl-root "$HARL" \
  --experiment-id TSP-MARL-MULTITASK-PHASE-A-001 \
  --results-root "$OUT" \
  --exp-name "stage_a_${TASK_ID}_${RAY}_${COUPLING}_q${Q}" \
  "${TASK_ARGS[@]}" \
  --coupling "$COUPLING" \
  --q "$Q" \
  --message-budget "$MESSAGE_BUDGET" \
  --environment-budget 1000000 \
  --actor-lr 0.0005 \
  --critic-lr 0.0005 \
  --seed "$SEED" \
  --seed-registry-base 112001 \
  --seed-registry-size 128 \
  --eval-threads 4 \
  --torch-threads 8 \
  --cuda

find "$OUT" -type f ! -name SHA256SUMS -print0 | sort -z | \
  xargs -0 sha256sum > "$OUT/SHA256SUMS"
