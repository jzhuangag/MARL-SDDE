#!/bin/bash
#SBATCH --job-name=cpf-pb-headroom
#SBATCH --partition=gpu-a30
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:25:00

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

codes=(signed signed_only cache mismatch_terminal complete_terminal age_prefix mismatch_prefix round_prefix static_prefix no_refresh full_refresh)
seeds=(79006 79007)
code_index=$((SLURM_ARRAY_TASK_ID % ${#codes[@]}))
seed_index=$((SLURM_ARRAY_TASK_ID / ${#codes[@]}))
code="${codes[$code_index]}"
seed="${seeds[$seed_index]}"

scheduler=signed_lyapunov
budget_rate=0.5
budget_enforcement=terminal
cone_shell=6
case "${code}" in
  signed) ;;
  signed_only) scheduler=signed_only ;;
  cache) scheduler=cache_lyapunov ;;
  mismatch_terminal) scheduler=mismatch ;;
  complete_terminal) scheduler=complete_burst ;;
  age_prefix) scheduler=age; budget_enforcement=prefix ;;
  mismatch_prefix) scheduler=mismatch; budget_enforcement=prefix ;;
  round_prefix) scheduler=round_robin; budget_enforcement=prefix ;;
  static_prefix) scheduler=static_chain; budget_enforcement=prefix ;;
  no_refresh) scheduler=no_refresh; budget_enforcement=prefix ;;
  full_refresh) scheduler=complete_burst; budget_rate=19; budget_enforcement=prefix; cone_shell=20 ;;
  *) printf 'unknown headroom code %s\n' "${code}" >&2; exit 4 ;;
esac

cd "${CPF_ROOT}/code"
test "$(git rev-parse HEAD)" = "${CPF_EXPECTED_COMMIT}"
source "${CPF_ROOT}/envs/pistonball-py310/bin/activate"
mkdir -p "${CPF_OUTPUT_ROOT}"
output="${CPF_OUTPUT_ROOT}/${seed}_${code}.json"
test ! -e "${output}"

python -m experiments.policy_dependency_sync.run_async_pistonball_training \
  --scheduler "${scheduler}" --seed "${seed}" --device cuda --output "${output}" \
  --launches 4096 --n-agents 20 --rollout-horizon 4 --max-cycles 125 \
  --batch-size 32 --evaluations 5 --evaluation-episodes 3 \
  --budget-rate "${budget_rate}" --budget-enforcement "${budget_enforcement}" \
  --queue-step 1 --cone-shell "${cone_shell}" --maximum-delay 8 --critic-step 0.0003 \
  --actor-receipt-step 0.01 --network-activation silu \
  --lyapunov-weight 100000000 --cache-debt-weight 83650.1230871728 \
  --random-drop --random-rotate

python - "${output}" "${code}" <<'PY'
import json
import sys

path, code = sys.argv[1:]
result = json.load(open(path, encoding="utf-8"))
result["headroom_code"] = code
with open(path, "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
