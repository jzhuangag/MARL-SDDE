#!/bin/bash
#SBATCH --job-name=tsp-smac-ctrl-smoke
#SBATCH --account=vincentlau
#SBATCH --partition=gpu-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a30:1
#SBATCH --time=00:30:00

set -euo pipefail

ROOT=/scratch/jzhuangag/MARL-SDDE-TSP-SMACV2-DEV-001
BASE=/scratch/jzhuangag/MARL-SDDE-TwoClocks-20260902
PYTHON="$BASE/envs/runtime-py39/bin/python"
HARL="$BASE/external/HARL"
CODE="$ROOT/code/MARL-SDDE"
export SC2PATH="$BASE/external/StarCraftII"
export PYTHONPATH="$CODE/TSP/experiments:$BASE/external/SMACv2:$HARL"
export TMPDIR="$ROOT/tmp/controller-smoke-${SLURM_JOB_ID}"
export MPLCONFIGDIR="$ROOT/tmp/mpl-controller-smoke-${SLURM_JOB_ID}"
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$ROOT/artifacts/smoke" "$ROOT/logs"

OUT="$ROOT/artifacts/smoke/controller-job${SLURM_JOB_ID}-shared-seed80401"
"$PYTHON" "$CODE/TSP/experiments/run_mappo_probe_commit.py" \
  --harl-root "$HARL" \
  --experiment-id TSP-MARL-SMACV2-CONTROLLER-SMOKE-001 \
  --results-root "$OUT" \
  --env-name smacv2 \
  --map-name terran_10_vs_10 \
  --coupling shared \
  --seed 80401 \
  --probe-seed 80402 \
  --seed-registry-base 80401 \
  --seed-registry-size 128 \
  --probe-q 8 \
  --probe-blocks 4 \
  --candidate-q 1 8 \
  --correlation-delta 0.05 \
  --rollout-length 200 \
  --message-budget 14400 \
  --environment-budget 1600 \
  --server-overhead 800 \
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

"$PYTHON" -c 'import json,numpy as np,sys; root=sys.argv[1]; x=np.load(root+"/probe_fingerprints.npy"); m=json.load(open(root+"/tsp_probe_commit_metadata.json")); assert x.shape==(4,8); assert np.max(np.ptp(x,axis=1))<=1e-10; assert m["decision"]["selected_q"]==1; print(json.dumps({"fingerprint_shape":list(x.shape),"max_within_block_range":float(np.max(np.ptp(x,axis=1))),"selected_q":m["decision"]["selected_q"]}))' "$OUT" \
  > "$OUT/controller_smoke_assertion.json"
find "$OUT" -type f ! -name SHA256SUMS -print0 | sort -z | \
  xargs -0 sha256sum > "$OUT/SHA256SUMS"
