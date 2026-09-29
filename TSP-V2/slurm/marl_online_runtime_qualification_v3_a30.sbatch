#!/bin/bash
#SBATCH --job-name=tspv2-qual3
#SBATCH --account=vincentlau
#SBATCH --partition=gpu-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:a30:1
#SBATCH --time=04:00:00
#SBATCH --array=0-7%2

set -euo pipefail

ROOT=/scratch/jzhuangag/MARL-SDDE-TSP-V2-ONLINE-QUAL-003
CODE="$ROOT/code/MARL-SDDE"
GLOBAL_INDEX=$SLURM_ARRAY_TASK_ID
TASK_INDEX=$((GLOBAL_INDEX / 4))
LOCAL_INDEX=$((GLOBAL_INDEX % 4))
COUPLING_INDEX=$((LOCAL_INDEX / 2))
METHOD_INDEX=$((LOCAL_INDEX % 2))

if [ "$COUPLING_INDEX" -eq 0 ]; then COUPLING=independent; else COUPLING=shared; fi
if [ "$METHOD_INDEX" -eq 0 ]; then METHOD=controller; else METHOD=fixed_q1; fi
SEED=$((997101+GLOBAL_INDEX))
VALIDATION_SEED_BASE=$((998101+20*GLOBAL_INDEX))
EVALUATION_SEED=$((999101+GLOBAL_INDEX))

COMMON=(
  --experiment-id TSP-V2-MARL-ONLINE-QUAL-003
  --development
  --coupling "$COUPLING"
  --method "$METHOD"
  --seed "$SEED"
  --validation-seed-base "$VALIDATION_SEED_BASE"
  --evaluation-seed "$EVALUATION_SEED"
  --seed-registry-base 997001
  --seed-registry-size 4096
  --candidate-q 1 2 4 8
  --window 8
  --bonus-scale 0.25
  --confidence-delta 0.05
  --markov-bias 0.0
  --probe-interval 40
  --progress-scale 0.005
  --actor-lr 0.0005
  --critic-lr 0.0005
  --torch-threads 8
  --eval-threads 1
  --eval-episodes 1
  --cuda
)

if [ "$TASK_INDEX" -eq 0 ]; then
  TASK_ID=mpe_simple_spread
  BASE=/scratch/jzhuangag/MARL-SDDE-TSP-MARL-CONF-001
  PYTHON="$BASE/venv/bin/python"
  HARL="$BASE/tmp/HARL"
  TASK_ARGS=(
    --env-name pettingzoo_mpe
    --scenario simple_spread_v2
    --no-continuous-actions
    --share-param
    --rollout-length 25
    --train-updates-per-decision 1
    --validation-q 1
    --validation-horizon 25
    --evaluation-horizon 25
    --evaluation-interval 9
    --validation-return-scale 5.0
    --controller-decisions 9
    --message-budget 5000
    --environment-budget 675
    --server-overhead 100
    --queue-weight 1000
    --hidden-size 64
    --ppo-epoch 2
    --critic-epoch 2
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
    --train-updates-per-decision 1
    --validation-q 1
    --validation-horizon 50
    --evaluation-horizon 1000
    --evaluation-interval 9
    --validation-return-scale 5.0
    --controller-decisions 9
    --message-budget 36900
    --environment-budget 2700
    --server-overhead 800
    --queue-weight 5000
    --hidden-size 128
    --ppo-epoch 5
    --critic-epoch 5
  )
fi

export PYTHONPATH="$CODE/TSP-V2/experiments:$CODE/TSP/experiments:$HARL"
export PYTHONNOUSERSITE=1
export TMPDIR="$ROOT/tmp/${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export MPLCONFIGDIR="$ROOT/tmp/mpl-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
OUT="$ROOT/artifacts/${TASK_ID}-${COUPLING}-${METHOD}-seed${SEED}"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$ROOT/artifacts" "$ROOT/logs"

"$PYTHON" "$CODE/TSP-V2/experiments/run_marl_online_drift_queue_v3.py" \
  --harl-root "$HARL" \
  --results-root "$OUT" \
  "${COMMON[@]}" \
  "${TASK_ARGS[@]}"

find "$OUT" -type f ! -name SHA256SUMS -print0 | sort -z | \
  xargs -0 sha256sum > "$OUT/SHA256SUMS"
