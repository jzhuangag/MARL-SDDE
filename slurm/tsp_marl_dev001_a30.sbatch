#!/usr/bin/env bash
#SBATCH --job-name=tsp-marl-d0
#SBATCH --partition=gpu-a30
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/jzhuangag/MARL-SDDE-TSP-MARL-DEV-001/logs/%x-%A_%a.out
#SBATCH --error=/scratch/jzhuangag/MARL-SDDE-TSP-MARL-DEV-001/logs/%x-%A_%a.err

set -euo pipefail

: "${TSP_EXPECTED_COMMIT:?set TSP_EXPECTED_COMMIT}"
: "${SLURM_ARRAY_TASK_ID:?submit this script as an array}"

ROOT=/scratch/jzhuangag/MARL-SDDE-TSP-MARL-DEV-001
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

couplings=(independent shared)
qs=(1 4 8 16)
lrs=(0.00025 0.0005)
task=${SLURM_ARRAY_TASK_ID}
if (( task < 0 || task >= 16 )); then
  echo "array index must be in 0..15" >&2
  exit 6
fi
coupling=${couplings[$((task / 8))]}
remainder=$((task % 8))
q=${qs[$((remainder / 2))]}
critic_lr=${lrs[$((remainder % 2))]}

CELL_ROOT=${ROOT}/artifacts/d0-${coupling}-q${q}-lr${critic_lr}
if [[ -e "${CELL_ROOT}" ]]; then
  echo "refusing to overwrite ${CELL_ROOT}" >&2
  exit 7
fi
mkdir -p "${CELL_ROOT}"

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MPLCONFIGDIR=${ROOT}/mplconfig/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export XDG_CACHE_HOME=${ROOT}/cache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

echo "experiment=TSP-MARL-DEV-001"
echo "commit=${TSP_EXPECTED_COMMIT}"
echo "cell=${coupling},q=${q},critic_lr=${critic_lr},seed=93001"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

cd "${CODE}"
"${PYTHON}" TSP/experiments/run_mappo_return_bridge.py \
  --harl-root "${HARL}" \
  --results-root "${CELL_ROOT}" \
  --exp-name "d0_${coupling}_q${q}_lr${critic_lr}" \
  --coupling "${coupling}" --q "${q}" --critic-lr "${critic_lr}" \
  --message-budget 5000000 --environment-budget 1000000 \
  --server-overhead 100 --rollout-length 25 \
  --seed 93001 --seed-registry-base 93001 --seed-registry-size 17 \
  --eval-interval 1000 --eval-episodes 32 --eval-threads 4 \
  --hidden-size 64 --activation tanh --ppo-epoch 2 --critic-epoch 2 \
  --cuda

mapfile -t metadata_files < <(find "${CELL_ROOT}" -type f -name tsp_bridge_metadata.json -print)
if (( ${#metadata_files[@]} != 1 )); then
  echo "expected exactly one metadata file, found ${#metadata_files[@]}" >&2
  exit 8
fi
run_dir=$(dirname "${metadata_files[0]}")
test -s "${run_dir}/progress.txt"
(
  cd "${run_dir}"
  sha256sum progress.txt tsp_bridge_metadata.json > SHA256SUMS
  sha256sum -c SHA256SUMS
)
echo "run_dir=${run_dir}"
