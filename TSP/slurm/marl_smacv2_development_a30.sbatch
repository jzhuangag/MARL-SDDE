#!/bin/bash
#SBATCH --job-name=tsp-smac-dev1
#SBATCH --account=vincentlau
#SBATCH --partition=gpu-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a30:1
#SBATCH --time=12:00:00
#SBATCH --array=0-23%4

set -euo pipefail

ROOT=/scratch/jzhuangag/MARL-SDDE-TSP-SMACV2-DEV-001
BASE=/scratch/jzhuangag/MARL-SDDE-TwoClocks-20260902
PYTHON="$BASE/envs/runtime-py39/bin/python"
HARL="$BASE/external/HARL"
CODE="$ROOT/code/MARL-SDDE"
export SC2PATH="$BASE/external/StarCraftII"
export PYTHONPATH="$CODE/TSP/experiments:$BASE/external/SMACv2:$HARL"
export TMPDIR="$ROOT/tmp/dev-${SLURM_ARRAY_TASK_ID}"
export MPLCONFIGDIR="$ROOT/tmp/mpl-dev-${SLURM_ARRAY_TASK_ID}"
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$ROOT/artifacts/development" "$ROOT/logs"

METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / 8))
REGIME_INDEX=$(((SLURM_ARRAY_TASK_ID % 8) / 4))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 4))
SEED=$((80101 + SEED_INDEX))
PROBE_SEED=$((80201 + SEED_INDEX))
if [ "$REGIME_INDEX" -eq 0 ]; then
  COUPLING=independent
else
  COUPLING=shared
fi

COMMON=(
  --harl-root "$HARL"
  --experiment-id TSP-MARL-SMACV2-DEV-001
  --env-name smacv2
  --map-name terran_10_vs_10
  --coupling "$COUPLING"
  --rollout-length 200
  --message-budget 16000000
  --environment-budget 3000000
  --server-overhead 800
  --seed "$SEED"
  --seed-registry-base 80101
  --seed-registry-size 128
  --hidden-size 128
  --activation relu
  --actor-lr 0.0005
  --critic-lr 0.0005
  --ppo-epoch 5
  --critic-epoch 5
  --eval-interval 250
  --log-interval 250
  --eval-episodes 64
  --eval-threads 8
  --torch-threads 8
  --cuda
)

case "$METHOD_INDEX" in
  0)
    OUT="$ROOT/artifacts/development/controller-${COUPLING}-seed${SEED}"
    "$PYTHON" "$CODE/TSP/experiments/run_mappo_probe_commit.py" \
      --results-root "$OUT" \
      --probe-seed "$PROBE_SEED" \
      --probe-q 8 \
      --probe-blocks 64 \
      --candidate-q 1 8 \
      --correlation-delta 0.05 \
      "${COMMON[@]}"
    ;;
  1)
    OUT="$ROOT/artifacts/development/fixed-q1-${COUPLING}-seed${SEED}"
    "$PYTHON" "$CODE/TSP/experiments/run_mappo_return_bridge.py" \
      --results-root "$OUT" --exp-name "fixed_q1_${COUPLING}" --q 1 \
      "${COMMON[@]}"
    ;;
  2)
    OUT="$ROOT/artifacts/development/fixed-q8-${COUPLING}-seed${SEED}"
    "$PYTHON" "$CODE/TSP/experiments/run_mappo_return_bridge.py" \
      --results-root "$OUT" --exp-name "fixed_q8_${COUPLING}" --q 8 \
      "${COMMON[@]}"
    ;;
  *)
    echo "unexpected method index: $METHOD_INDEX" >&2
    exit 2
    ;;
esac

find "$OUT" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/SHA256SUMS"
