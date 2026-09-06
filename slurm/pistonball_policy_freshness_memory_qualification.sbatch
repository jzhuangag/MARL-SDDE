#!/bin/bash
#SBATCH --job-name=cpf-pb-mem
#SBATCH --partition=gpu-a30
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00

set -euo pipefail

: "${CPF_ROOT:?CPF_ROOT is required}"
: "${CPF_OUTPUT:?CPF_OUTPUT is required}"
: "${CPF_EXPECTED_COMMIT:?CPF_EXPECTED_COMMIT is required}"

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "${CPF_ROOT}/code"
test "$(git rev-parse HEAD)" = "${CPF_EXPECTED_COMMIT}"
source "${CPF_ROOT}/envs/pistonball-py310/bin/activate"
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'

mkdir -p "$(dirname "${CPF_OUTPUT}")"
/usr/bin/time -v python -m experiments.policy_dependency_sync.run_async_pistonball_training \
  --scheduler signed_lyapunov \
  --seed 78501 \
  --device cuda \
  --output "${CPF_OUTPUT}" \
  --launches 1024 \
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

result = json.load(open(sys.argv[1], encoding="utf-8"))
assert result["finite"] and result["budget_feasible"]
assert result["remaining_packets_after_drain"] == 0
actual_cycles = sum(row["segment_cycles"] for row in result["launch_trace"])
assert result["launched_actor_transitions"] == 20 * actual_cycles
assert 0 < result["launched_actor_transitions"] <= 1024 * 4 * 20
assert result["evaluations"][-1]["actor_transitions"] == result["launched_actor_transitions"]
assert result["received_packets"] == result["launched_gradient_packets"]
print(
    {
        "launched_actor_transitions": result["launched_actor_transitions"],
        "launched_gradient_packets": result["launched_gradient_packets"],
        "spent_refresh_units": result["spent_refresh_units"],
        "allowed_refresh_units": result["allowed_refresh_units"],
        "runtime_seconds": result["runtime_seconds"],
        "action_trace_sha256": result["action_trace_sha256"],
    }
)
PY
