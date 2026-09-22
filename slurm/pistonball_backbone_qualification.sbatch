#!/bin/bash
#SBATCH --job-name=cpf-pb-backbone
#SBATCH --partition=gpu-a30
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

: "${CPF_ROOT:?CPF_ROOT is required}"
: "${CPF_OUTPUT_ROOT:?CPF_OUTPUT_ROOT is required}"
: "${CPF_EXPECTED_COMMIT:?CPF_EXPECTED_COMMIT is required}"
: "${CPF_CONFIGS:?CPF_CONFIGS is required}"

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "${CPF_ROOT}/code"
test "$(git rev-parse HEAD)" = "${CPF_EXPECTED_COMMIT}"
IFS=':' read -r -a config_array <<< "${CPF_CONFIGS}"
code="${config_array[${SLURM_ARRAY_TASK_ID}]}"

scheduler=complete_burst
budget_rate=19
cone_shell=20
critic_step=0.0003
actor_step=0.001
case "${code}" in
  fresh_s1) ;;
  fresh_s3) actor_step=0.003 ;;
  fresh_s10) actor_step=0.01 ;;
  fresh_c1_s3) critic_step=0.001; actor_step=0.003 ;;
  no_s3) scheduler=no_refresh; budget_rate=0; cone_shell=6; critic_step=0.001; actor_step=0.003 ;;
  *) printf 'unknown qualification code %s\n' "${code}" >&2; exit 5 ;;
esac

source "${CPF_ROOT}/envs/pistonball-py310/bin/activate"
mkdir -p "${CPF_OUTPUT_ROOT}"
output="${CPF_OUTPUT_ROOT}/${code}.json"
test ! -e "${output}"

python -m experiments.policy_dependency_sync.run_async_pistonball_training \
  --scheduler "${scheduler}" \
  --seed 79003 \
  --device cuda \
  --output "${output}" \
  --launches 4096 \
  --n-agents 20 \
  --rollout-horizon 4 \
  --max-cycles 125 \
  --batch-size 32 \
  --evaluations 5 \
  --evaluation-episodes 3 \
  --budget-rate "${budget_rate}" \
  --cone-shell "${cone_shell}" \
  --maximum-delay 0 \
  --critic-step "${critic_step}" \
  --actor-receipt-step "${actor_step}" \
  --lyapunov-weight 100000 \
  --cache-debt-weight 100000 \
  --random-drop \
  --random-rotate

python - "${output}" "${code}" <<'PY'
import json
import math
import sys

path, code = sys.argv[1:]
result = json.load(open(path, encoding="utf-8"))
result["qualification_code"] = code
assert result["device"] == "cuda"
assert result["deterministic_algorithms"]
assert result["finite"] and result["budget_feasible"]
assert result["remaining_packets_after_drain"] == 0
assert result["launched_actor_transitions"] == 4096 * 4 * 20
assert result["received_packets"] == result["launched_gradient_packets"]
assert result["critic_updates"] == result["launched_gradient_packets"]
assert all(math.isfinite(value) for value in result["actor_parameter_drift_l2"])
with open(path, "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
