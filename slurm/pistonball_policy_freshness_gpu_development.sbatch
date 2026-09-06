#!/bin/bash
#SBATCH --job-name=cpf-pb-dev
#SBATCH --partition=gpu-a30
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

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

cd "${CPF_ROOT}/code"
actual_commit="$(git rev-parse HEAD)"
if [[ "${actual_commit}" != "${CPF_EXPECTED_COMMIT}" ]]; then
  printf 'commit mismatch: expected %s actual %s\n' \
    "${CPF_EXPECTED_COMMIT}" "${actual_commit}" >&2
  exit 4
fi

IFS=':' read -r -a scheduler_array <<< "${CPF_SCHEDULERS}"
scheduler="${scheduler_array[${SLURM_ARRAY_TASK_ID}]}"
if [[ -z "${scheduler}" ]]; then
  printf 'no scheduler at array index %s\n' "${SLURM_ARRAY_TASK_ID}" >&2
  exit 5
fi

source "${CPF_ROOT}/envs/pistonball-py310/bin/activate"
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'

mkdir -p "${CPF_OUTPUT_ROOT}"
output="${CPF_OUTPUT_ROOT}/${scheduler}.json"
if [[ -e "${output}" ]]; then
  printf 'refuse to overwrite %s\n' "${output}" >&2
  exit 6
fi

python -m experiments.policy_dependency_sync.run_async_pistonball_training \
  --scheduler "${scheduler}" \
  --seed 79001 \
  --device cuda \
  --output "${output}" \
  --launches 4096 \
  --n-agents 20 \
  --rollout-horizon 4 \
  --max-cycles 125 \
  --batch-size 32 \
  --evaluations 9 \
  --evaluation-episodes 3 \
  --budget-rate 0.5 \
  --cone-shell 6 \
  --maximum-delay 4 \
  --lyapunov-weight 100000 \
  --cache-debt-weight 100000 \
  --random-drop \
  --random-rotate

python - "${output}" "${scheduler}" <<'PY'
import json
import sys

path, scheduler = sys.argv[1:]
result = json.load(open(path, encoding="utf-8"))
assert result["scheduler"] == scheduler
assert result["device"] == "cuda"
assert result["finite"]
assert result["budget_feasible"]
assert result["remaining_packets_after_drain"] == 0
assert result["launched_actor_transitions"] == 4096 * 4 * 20
assert result["received_packets"] == 4096 - (32 // 4) + 1
print(
    {
        "scheduler": scheduler,
        "initial_return": result["evaluations"][0]["return"],
        "terminal_return": result["evaluations"][-1]["return"],
        "spent_refresh_units": result["spent_refresh_units"],
        "allowed_refresh_units": result["allowed_refresh_units"],
        "runtime_seconds": result["runtime_seconds"],
        "action_trace_sha256": result["action_trace_sha256"],
    }
)
PY
