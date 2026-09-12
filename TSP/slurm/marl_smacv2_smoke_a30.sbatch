#!/bin/bash
#SBATCH --job-name=tsp-smac-smoke
#SBATCH --account=vincentlau
#SBATCH --partition=gpu-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a30:1
#SBATCH --time=00:30:00
#SBATCH --array=0-1%2

set -euo pipefail

ROOT=/scratch/jzhuangag/MARL-SDDE-TSP-SMACV2-DEV-001
BASE=/scratch/jzhuangag/MARL-SDDE-TwoClocks-20260902
PYTHON="$BASE/envs/runtime-py39/bin/python"
HARL="$BASE/external/HARL"
export SC2PATH="$BASE/external/StarCraftII"
export PYTHONPATH="$ROOT/code/MARL-SDDE/TSP/experiments:$BASE/external/SMACv2:$HARL"
export TMPDIR="$ROOT/tmp/smoke-${SLURM_ARRAY_TASK_ID}"
export MPLCONFIGDIR="$ROOT/tmp/mpl-${SLURM_ARRAY_TASK_ID}"
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$ROOT/artifacts/smoke" "$ROOT/logs"

case "$SLURM_ARRAY_TASK_ID" in
  0)
    COUPLING=independent
    Q=8
    SEED=112001
    ;;
  1)
    COUPLING=shared
    Q=1
    SEED=112002
    ;;
  *)
    echo "unexpected array index: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

OUT="$ROOT/artifacts/smoke/${COUPLING}-q${Q}-seed${SEED}"
"$PYTHON" "$ROOT/code/MARL-SDDE/TSP/experiments/run_mappo_return_bridge.py" \
  --harl-root "$HARL" \
  --experiment-id TSP-MARL-SMACV2-SMOKE-001 \
  --results-root "$OUT" \
  --exp-name "smoke_${COUPLING}_q${Q}" \
  --env-name smacv2 \
  --map-name terran_10_vs_10 \
  --coupling "$COUPLING" \
  --q "$Q" \
  --rollout-length 200 \
  --message-budget 4800 \
  --environment-budget 400 \
  --server-overhead 800 \
  --seed "$SEED" \
  --seed-registry-base 112001 \
  --seed-registry-size 128 \
  --hidden-size 128 \
  --activation relu \
  --ppo-epoch 5 \
  --critic-epoch 5 \
  --eval-interval 1 \
  --log-interval 1 \
  --eval-episodes 2 \
  --eval-threads 1 \
  --torch-threads 8 \
  --cuda

find "$OUT" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/SHA256SUMS"
