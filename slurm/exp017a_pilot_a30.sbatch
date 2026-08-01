#!/usr/bin/env bash
#SBATCH --job-name=exp017a-nonlinear-pilot
#SBATCH --partition=gpu-a30
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/jzhuangag/exp017a_logs/%x-%A_%a.out
#SBATCH --error=/scratch/jzhuangag/exp017a_logs/%x-%A_%a.err

set -euo pipefail

: "${EXP017A_CODE_ROOT:?set EXP017A_CODE_ROOT}"
: "${EXP017A_RUN_ROOT:?set EXP017A_RUN_ROOT}"
: "${EXP017A_PYTHON:?set EXP017A_PYTHON}"
: "${EXP017A_EXPECTED_COMMIT:?set EXP017A_EXPECTED_COMMIT}"

case "${EXP017A_CODE_ROOT}" in
  /scratch/jzhuangag/*) ;;
  *) echo "code root must be under /scratch/jzhuangag" >&2; exit 2 ;;
esac
case "${EXP017A_RUN_ROOT}" in
  /scratch/jzhuangag/*) ;;
  *) echo "run root must be under /scratch/jzhuangag" >&2; exit 2 ;;
esac

observed_commit="$(git -C "${EXP017A_CODE_ROOT}" rev-parse HEAD)"
if [[ "${observed_commit}" != "${EXP017A_EXPECTED_COMMIT}" ]]; then
  echo "commit mismatch: ${observed_commit}" >&2
  exit 3
fi
if [[ -n "$(git -C "${EXP017A_CODE_ROOT}" status --porcelain)" ]]; then
  echo "remote code worktree is dirty" >&2
  exit 4
fi

seeds=(20550101 20550102)
seed="${seeds[${SLURM_ARRAY_TASK_ID}]}"
seed_dir="${EXP017A_RUN_ROOT}/seeds/${seed}"
if [[ -e "${seed_dir}" ]]; then
  echo "refusing to overwrite ${seed_dir}" >&2
  exit 5
fi

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="${EXP017A_RUN_ROOT}/mplconfig-${SLURM_ARRAY_TASK_ID}"
export XDG_CACHE_HOME="${EXP017A_RUN_ROOT}/cache-${SLURM_ARRAY_TASK_ID}"

cd "${EXP017A_CODE_ROOT}/experiments/nonlinear_markov_td"
"${EXP017A_PYTHON}" run_exp017a_nonlinear_pilot.py \
  --seed "${seed}" \
  --device cuda \
  --output-dir "${seed_dir}"

(
  cd "${seed_dir}"
  sha256sum endpoints.csv trajectories.csv metadata.json > SHA256SUMS
  sha256sum -c SHA256SUMS
)
