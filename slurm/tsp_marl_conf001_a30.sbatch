#!/usr/bin/env bash
#SBATCH --job-name=tsp-marl-conf1
#SBATCH --partition=gpu-a30
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/jzhuangag/MARL-SDDE-TSP-MARL-CONF-001/logs/%x-%A_%a.out
#SBATCH --error=/scratch/jzhuangag/MARL-SDDE-TSP-MARL-CONF-001/logs/%x-%A_%a.err

set -euo pipefail

: "${TSP_EXPECTED_COMMIT:?set TSP_EXPECTED_COMMIT}"
: "${SLURM_ARRAY_TASK_ID:?submit this script as an array}"

ROOT=/scratch/jzhuangag/MARL-SDDE-TSP-MARL-CONF-001
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
if [[ -n "$(git -C "${HARL}" status --porcelain)" ]]; then
  echo "HARL worktree is dirty" >&2
  exit 5
fi

task=${SLURM_ARRAY_TASK_ID}
if (( task < 0 || task >= 48 )); then
  echo "array index must be in 0..47" >&2
  exit 6
fi

training_seeds=(96001 96002 96003 96004 96005 96006 96007 96008)
probe_seeds=(97001 97002 97003 97004 97005 97006 97007 97008)
couplings=(independent shared)

if (( task < 16 )); then
  method=controller
  local_index=${task}
  q=0
elif (( task < 32 )); then
  method=fixed_q1
  local_index=$((task - 16))
  q=1
else
  method=fixed_q8
  local_index=$((task - 32))
  q=8
fi
coupling=${couplings[$((local_index / 8))]}
seed_index=$((local_index % 8))
seed=${training_seeds[${seed_index}]}
probe_seed=${probe_seeds[${seed_index}]}
CELL_ROOT=${ROOT}/artifacts/${method}-${coupling}-seed${seed}

if [[ -e "${CELL_ROOT}" ]]; then
  echo "refusing to overwrite ${CELL_ROOT}" >&2
  exit 7
fi

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MPLCONFIGDIR=${ROOT}/mplconfig/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export XDG_CACHE_HOME=${ROOT}/cache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

cd "${CODE}"
if [[ "${method}" == "controller" ]]; then
  "${PYTHON}" TSP/experiments/run_mappo_probe_commit.py \
    --harl-root "${HARL}" --results-root "${CELL_ROOT}" \
    --coupling "${coupling}" --seed "${seed}" --probe-seed "${probe_seed}" \
    --probe-q 8 --probe-blocks 128 --candidate-q 1 8 \
    --correlation-delta 0.05 --critic-lr 0.0005 --actor-lr 0.0005 \
    --message-budget 5000000 --environment-budget 1000000 \
    --server-overhead 100 --rollout-length 25 \
    --seed-registry-base 90000 --seed-registry-size 10000 \
    --eval-interval 1000 --eval-episodes 32 --eval-threads 4 \
    --hidden-size 64 --activation tanh --ppo-epoch 2 --critic-epoch 2 --cuda
  test -s "${CELL_ROOT}/tsp_probe_commit_metadata.json"
  test -s "${CELL_ROOT}/charged_progress.csv"
  test -s "${CELL_ROOT}/probe_fingerprints.npy"
  (
    cd "${CELL_ROOT}"
    sha256sum tsp_probe_commit_metadata.json charged_progress.csv probe_fingerprints.npy > SHA256SUMS
    sha256sum -c SHA256SUMS
  )
else
  mkdir -p "${CELL_ROOT}"
  "${PYTHON}" TSP/experiments/run_mappo_return_bridge.py \
    --harl-root "${HARL}" --results-root "${CELL_ROOT}" \
    --exp-name "${method}_${coupling}_seed${seed}" \
    --coupling "${coupling}" --q "${q}" \
    --critic-lr 0.0005 --actor-lr 0.0005 \
    --message-budget 5000000 --environment-budget 1000000 \
    --server-overhead 100 --rollout-length 25 \
    --seed "${seed}" --seed-registry-base 90000 --seed-registry-size 10000 \
    --eval-interval 1000 --eval-episodes 32 --eval-threads 4 \
    --hidden-size 64 --activation tanh --ppo-epoch 2 --critic-epoch 2 --cuda
  mapfile -t metadata_files < <(find "${CELL_ROOT}" -type f -name tsp_bridge_metadata.json -print)
  if (( ${#metadata_files[@]} != 1 )); then
    echo "expected one fixed-run metadata file" >&2
    exit 8
  fi
  run_dir=$(dirname "${metadata_files[0]}")
  test -s "${run_dir}/progress.txt"
  (
    cd "${run_dir}"
    sha256sum progress.txt tsp_bridge_metadata.json > SHA256SUMS
    sha256sum -c SHA256SUMS
  )
fi

