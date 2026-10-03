#!/bin/bash
#SBATCH --job-name=tsp-smac-headroom
#SBATCH --account=vincentlau
#SBATCH --partition=gpu-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a30:1
#SBATCH --time=16:00:00
#SBATCH --array=0-7%4

set -euo pipefail

ROOT=/scratch/jzhuangag/MARL-SDDE-TSP-SMACV2-DEV-001
BASE=/scratch/jzhuangag/MARL-SDDE-TwoClocks-20260902
PYTHON="$BASE/envs/runtime-py39/bin/python"
HARL="$BASE/external/HARL"
CODE="$ROOT/code/MARL-SDDE"
export SC2PATH="$BASE/external/StarCraftII"
export PYTHONPATH="$CODE/TSP/experiments:$BASE/external/SMACv2:$HARL"
export TMPDIR="$ROOT/tmp/headroom-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export MPLCONFIGDIR="$ROOT/tmp/mpl-headroom-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$ROOT/artifacts/headroom" "$ROOT/logs"

Q_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
REGIME_INDEX=$(((SLURM_ARRAY_TASK_ID % 4) / 2))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 2))
if [ "$Q_INDEX" -eq 0 ]; then Q=1; else Q=8; fi
if [ "$REGIME_INDEX" -eq 0 ]; then COUPLING=independent; else COUPLING=shared; fi
SEED=$((81101+SEED_INDEX))
OUT="$ROOT/artifacts/headroom/fixed-q${Q}-${COUPLING}-seed${SEED}"

"$PYTHON" "$CODE/TSP/experiments/run_mappo_return_bridge.py" \
  --harl-root "$HARL" \
  --experiment-id TSP-MARL-SMACV2-HEADROOM-001 \
  --results-root "$OUT" \
  --exp-name "headroom_fixed_q${Q}_${COUPLING}" \
  --env-name smacv2 \
  --map-name terran_10_vs_10 \
  --coupling "$COUPLING" \
  --q "$Q" \
  --rollout-length 200 \
  --message-budget 6400000 \
  --environment-budget 1200000 \
  --server-overhead 800 \
  --seed "$SEED" \
  --seed-registry-base 81101 \
  --seed-registry-size 128 \
  --hidden-size 128 \
  --activation relu \
  --actor-lr 0.0005 \
  --critic-lr 0.0005 \
  --ppo-epoch 5 \
  --critic-epoch 5 \
  --eval-interval 200 \
  --log-interval 200 \
  --eval-episodes 32 \
  --eval-threads 8 \
  --torch-threads 8 \
  --cuda

find "$OUT" -type f ! -name SHA256SUMS -print0 | sort -z | \
  xargs -0 sha256sum > "$OUT/SHA256SUMS"
