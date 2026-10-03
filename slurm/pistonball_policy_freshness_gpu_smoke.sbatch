#!/bin/bash
#SBATCH --job-name=cpf-pb-smoke
#SBATCH --partition=gpu-a30
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

: "${CPF_ROOT:?CPF_ROOT is required}"
: "${CPF_OUTPUT:?CPF_OUTPUT is required}"
: "${CPF_EXPECTED_COMMIT:?CPF_EXPECTED_COMMIT is required}"

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "${CPF_ROOT}/code"
actual_commit="$(git rev-parse HEAD)"
if [[ "${actual_commit}" != "${CPF_EXPECTED_COMMIT}" ]]; then
  printf 'commit mismatch: expected %s actual %s\n' \
    "${CPF_EXPECTED_COMMIT}" "${actual_commit}" >&2
  exit 4
fi

source "${CPF_ROOT}/envs/pistonball-py310/bin/activate"
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

mkdir -p "$(dirname "${CPF_OUTPUT}")"
python -m experiments.policy_dependency_sync.run_async_pistonball_training \
  --scheduler signed_lyapunov \
  --seed 78001 \
  --device cuda \
  --output "${CPF_OUTPUT}" \
  --launches 64 \
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

python - "${CPF_OUTPUT}" <<'PY'
import json
import sys

path = sys.argv[1]
result = json.load(open(path, encoding="utf-8"))
assert result["device"] == "cuda"
assert result["deterministic_algorithms"]
assert result["finite"]
assert result["budget_feasible"]
assert result["remaining_packets_after_drain"] == 0
assert result["launched_actor_transitions"] == 64 * 4 * 20
assert result["received_packets"] == result["launched_gradient_packets"]
print(
    {
        "spent_refresh_units": result["spent_refresh_units"],
        "allowed_refresh_units": result["allowed_refresh_units"],
        "optional_policy_bytes": result["optional_policy_bytes"],
        "runtime_seconds": result["runtime_seconds"],
        "action_trace_sha256": result["action_trace_sha256"],
    }
)
PY
