#!/bin/bash
#SBATCH --job-name=cpf-pb-placebo
#SBATCH --partition=gpu-a30
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00

set -euo pipefail

: "${CPF_ROOT:?CPF_ROOT is required}"
: "${CPF_OUTPUT_ROOT:?CPF_OUTPUT_ROOT is required}"
: "${CPF_EXPECTED_COMMIT:?CPF_EXPECTED_COMMIT is required}"
: "${CPF_SCHEDULERS:?CPF_SCHEDULERS is required}"

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "${CPF_ROOT}/code"
test "$(git rev-parse HEAD)" = "${CPF_EXPECTED_COMMIT}"
IFS=':' read -r -a scheduler_array <<< "${CPF_SCHEDULERS}"
scheduler="${scheduler_array[${SLURM_ARRAY_TASK_ID}]}"
test "${scheduler}" = no_refresh || test "${scheduler}" = signed_only

source "${CPF_ROOT}/envs/pistonball-py310/bin/activate"
mkdir -p "${CPF_OUTPUT_ROOT}"
output="${CPF_OUTPUT_ROOT}/${scheduler}.json"
test ! -e "${output}"

python -m experiments.policy_dependency_sync.run_async_pistonball_training \
  --scheduler "${scheduler}" \
  --seed 79002 \
  --device cuda \
  --output "${output}" \
  --launches 512 \
  --n-agents 20 \
  --rollout-horizon 4 \
  --max-cycles 125 \
  --batch-size 32 \
  --evaluations 2 \
  --evaluation-episodes 1 \
  --budget-rate 0.5 \
  --cone-shell 6 \
  --maximum-delay 4 \
  --lyapunov-weight 100000 \
  --cache-debt-weight 100000 \
  --random-drop \
  --random-rotate

python - "${output}" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
assert result["device"] == "cuda"
assert result["deterministic_algorithms"]
assert result["finite"] and result["budget_feasible"]
assert result["remaining_packets_after_drain"] == 0
assert result["launched_actor_transitions"] == 512 * 4 * 20
assert result["received_packets"] == result["launched_gradient_packets"]
PY
