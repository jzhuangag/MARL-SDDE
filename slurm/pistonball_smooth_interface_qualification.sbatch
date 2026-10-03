#!/bin/bash
#SBATCH --job-name=cpf-pb-smooth
#SBATCH --partition=gpu-a30
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00

set -euo pipefail

: "${CPF_ROOT:?CPF_ROOT is required}"
: "${CPF_OUTPUT_ROOT:?CPF_OUTPUT_ROOT is required}"
: "${CPF_EXPECTED_COMMIT:?CPF_EXPECTED_COMMIT is required}"

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "${CPF_ROOT}/code"
test "$(git rev-parse HEAD)" = "${CPF_EXPECTED_COMMIT}"
source "${CPF_ROOT}/envs/pistonball-py310/bin/activate"
mkdir -p "${CPF_OUTPUT_ROOT}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    output="${CPF_OUTPUT_ROOT}/fresh_silu.json"
    test ! -e "${output}"
    python -m experiments.policy_dependency_sync.run_async_pistonball_training \
      --scheduler complete_burst --seed 79005 --device cuda --output "${output}" \
      --launches 4096 --n-agents 20 --rollout-horizon 4 --max-cycles 125 \
      --batch-size 32 --evaluations 5 --evaluation-episodes 3 \
      --budget-rate 19 --budget-enforcement prefix --queue-step 1 \
      --cone-shell 20 --maximum-delay 0 --critic-step 0.0003 \
      --actor-receipt-step 0.01 --network-activation silu \
      --lyapunov-weight 100000 --cache-debt-weight 100000 \
      --random-drop --random-rotate
    ;;
  1)
    output="${CPF_OUTPUT_ROOT}/scale_silu.json"
    test ! -e "${output}"
    python -m experiments.policy_dependency_sync.run_async_pistonball_training \
      --scheduler signed_lyapunov --seed 79004 --device cuda --output "${output}" \
      --launches 2048 --n-agents 20 --rollout-horizon 4 --max-cycles 125 \
      --batch-size 32 --evaluations 3 --evaluation-episodes 2 \
      --budget-rate 0.5 --budget-enforcement terminal --queue-step 1 \
      --cone-shell 6 --maximum-delay 8 --critic-step 0.0003 \
      --actor-receipt-step 0.01 --network-activation silu \
      --lyapunov-weight 100000 --cache-debt-weight 100000 \
      --random-drop --random-rotate
    ;;
  *) printf 'unknown array task %s\n' "${SLURM_ARRAY_TASK_ID}" >&2; exit 4 ;;
esac
