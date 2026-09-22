#!/usr/bin/env bash
#SBATCH --job-name=tsp-xtask-smoke
#SBATCH --partition=gpu-a30
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/jzhuangag/MARL-SDDE-TSP-MARL-XTASK-SMOKE-001/logs/%x-%A_%a.out
#SBATCH --error=/scratch/jzhuangag/MARL-SDDE-TSP-MARL-XTASK-SMOKE-001/logs/%x-%A_%a.err

set -euo pipefail

: "${TSP_EXPECTED_COMMIT:?set TSP_EXPECTED_COMMIT}"
: "${SLURM_ARRAY_TASK_ID:?submit this script as a two-cell array}"

ROOT=/scratch/jzhuangag/MARL-SDDE-TSP-MARL-XTASK-SMOKE-001
CODE=${ROOT}/code
HARL=${ROOT}/tmp/HARL
PYTHON=${ROOT}/venv/bin/python

if [[ "$(git -C "${CODE}" rev-parse HEAD)" != "${TSP_EXPECTED_COMMIT}" ]]; then
  echo "code commit mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "${CODE}" status --porcelain)" ]]; then
  echo "code worktree is dirty" >&2
  exit 3
fi
if [[ "$(git -C "${HARL}" rev-parse HEAD)" != "b1af98b0dbab72a2eee9d160751cd09aedbb8ce2" ]]; then
  echo "HARL commit mismatch" >&2
  exit 4
fi

couplings=(independent shared)
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= 2 )); then
  echo "smoke array index must be 0 or 1" >&2
  exit 5
fi
coupling=${couplings[${SLURM_ARRAY_TASK_ID}]}
seed=$((105991 + SLURM_ARRAY_TASK_ID))
probe_seed=$((105993 + SLURM_ARRAY_TASK_ID))
CELL_ROOT=${ROOT}/artifacts/controller-${coupling}
if [[ -e "${CELL_ROOT}" ]]; then
  echo "refusing to overwrite ${CELL_ROOT}" >&2
  exit 6
fi

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MPLCONFIGDIR=${ROOT}/mplconfig/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export XDG_CACHE_HOME=${ROOT}/cache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

cd "${CODE}"
"${PYTHON}" TSP/experiments/run_mappo_probe_commit.py \
  --experiment-id TSP-MARL-XTASK-SMOKE-001 \
  --harl-root "${HARL}" --results-root "${CELL_ROOT}" \
  --scenario simple_reference_v2 --coupling "${coupling}" \
  --seed "${seed}" --probe-seed "${probe_seed}" \
  --probe-q 8 --probe-blocks 8 --candidate-q 1 8 \
  --correlation-delta 0.05 --critic-lr 0.0005 --actor-lr 0.0005 \
  --message-budget 20000 --environment-budget 5000 \
  --server-overhead 100 --rollout-length 25 \
  --seed-registry-base 100000 --seed-registry-size 10000 \
  --eval-interval 5 --eval-episodes 2 --eval-threads 2 \
  --log-interval 5 --hidden-size 64 --activation tanh \
  --ppo-epoch 2 --critic-epoch 2 --cuda

test -s "${CELL_ROOT}/tsp_probe_commit_metadata.json"
test -s "${CELL_ROOT}/charged_progress.csv"
test -s "${CELL_ROOT}/probe_fingerprints.npy"
(
  cd "${CELL_ROOT}"
  sha256sum tsp_probe_commit_metadata.json charged_progress.csv probe_fingerprints.npy > SHA256SUMS
  sha256sum -c SHA256SUMS
)
