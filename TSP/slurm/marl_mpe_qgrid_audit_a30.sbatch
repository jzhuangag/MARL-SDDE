#!/bin/bash
#SBATCH --job-name=tsp-mpe-qgrid
#SBATCH --account=vincentlau
#SBATCH --partition=gpu-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a30:1
#SBATCH --time=04:00:00
#SBATCH --array=0-7%2

set -euo pipefail

GLOBAL_INDEX=$((${OFFSET:?OFFSET must be exported}+SLURM_ARRAY_TASK_ID))
ROOT=/scratch/jzhuangag/MARL-SDDE-TSP-MPE-QGRID-AUDIT-001
REFERENCE=/scratch/jzhuangag/MARL-SDDE-TSP-MARL-CONF-001
PYTHON="$REFERENCE/venv/bin/python"
HARL="$REFERENCE/tmp/HARL"
CODE="$ROOT/code/MARL-SDDE"
export PYTHONPATH="$CODE/TSP/experiments:$HARL"
export PYTHONNOUSERSITE=1
export TMPDIR="$ROOT/tmp/${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export MPLCONFIGDIR="$ROOT/tmp/mpl-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$ROOT/artifacts" "$ROOT/logs"

Q_INDEX=$((GLOBAL_INDEX / 16))
REGIME_INDEX=$(((GLOBAL_INDEX % 16) / 8))
SEED_INDEX=$((GLOBAL_INDEX % 8))
if [ "$Q_INDEX" -eq 0 ]; then Q=2; else Q=4; fi
if [ "$REGIME_INDEX" -eq 0 ]; then COUPLING=independent; else COUPLING=shared; fi
SEED=$((96001+SEED_INDEX))
OUT="$ROOT/artifacts/${ARTIFACT_SET:?ARTIFACT_SET must be exported}/fixed-q${Q}-${COUPLING}-seed${SEED}"

"$PYTHON" "$CODE/TSP/experiments/run_mappo_return_bridge.py" \
  --harl-root "$HARL" \
  --experiment-id TSP-MARL-MPE-QGRID-AUDIT-001 \
  --results-root "$OUT" \
  --exp-name "qgrid_fixed_q${Q}_${COUPLING}" \
  --env-name pettingzoo_mpe \
  --scenario simple_spread_v2 \
  --coupling "$COUPLING" \
  --q "$Q" \
  --rollout-length 25 \
  --message-budget 5000000 \
  --environment-budget 1000000 \
  --server-overhead 100 \
  --seed "$SEED" \
  --seed-registry-base 96001 \
  --seed-registry-size 128 \
  --hidden-size 64 \
  --activation tanh \
  --actor-lr 0.0005 \
  --critic-lr 0.0005 \
  --ppo-epoch 2 \
  --critic-epoch 2 \
  --eval-interval 1000 \
  --log-interval 1000 \
  --eval-episodes 32 \
  --eval-threads 4 \
  --torch-threads 8 \
  --cuda

find "$OUT" -type f ! -name SHA256SUMS -print0 | sort -z | \
  xargs -0 sha256sum > "$OUT/SHA256SUMS"
