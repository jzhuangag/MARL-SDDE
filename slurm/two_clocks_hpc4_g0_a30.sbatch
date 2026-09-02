#!/bin/bash
#SBATCH --job-name=tc-g0
#SBATCH --partition=gpu-a30
#SBATCH --account=vincentlau
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a30:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/jzhuangag/MARL-SDDE-TwoClocks-20260902/logs/two-clocks-g0-%j.out
#SBATCH --error=/scratch/jzhuangag/MARL-SDDE-TwoClocks-20260902/logs/two-clocks-g0-%j.err

set -euo pipefail

: "${EXPECTED_CODE_COMMIT:?submit with EXPECTED_CODE_COMMIT}"
ROOT=/scratch/jzhuangag/MARL-SDDE-TwoClocks-20260902
CODE_ROOT="$ROOT/code"
HARL_ROOT="$ROOT/external/HARL"
SMACV2_ROOT="$ROOT/external/SMACv2"
RUNTIME="$ROOT/envs/runtime-py39"
RUN_DIR="$ROOT/results/g0-$SLURM_JOB_ID"

if [[ -e "$RUN_DIR" ]]; then
  echo "refusing to overwrite $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR" "$ROOT/plots" "$ROOT/provenance"

actual_commit=$(git -C "$CODE_ROOT" rev-parse HEAD)
[[ "$actual_commit" == "$EXPECTED_CODE_COMMIT" ]]
[[ -z "$(git -C "$CODE_ROOT" status --short)" ]]

export PYTHONNOUSERSITE=1
export PATH="$RUNTIME/bin:$PATH"
export PYTHONPATH="$CODE_ROOT:$HARL_ROOT:$SMACV2_ROOT"
export MUJOCO_PY_MUJOCO_PATH="$ROOT/external/mujoco210"
export SC2PATH="$ROOT/external/StarCraftII"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:/usr/lib/nvidia:/usr/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="$ROOT/external/glew-2.2.0-release/include:${CPATH:-}"
export CFLAGS="${CFLAGS:-} -DGLEW_NO_GLU"
export XDG_CACHE_HOME="$ROOT/envs/cache"
export MPLCONFIGDIR="$ROOT/envs/matplotlib"
export TORCH_HOME="$ROOT/envs/torch-cache"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES

{
  echo "job_id=$SLURM_JOB_ID"
  echo "host=$(hostname)"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "code_commit=$actual_commit"
  echo "harl_commit=$(git -C "$HARL_ROOT" rev-parse HEAD)"
  echo "smacv2_commit=$(git -C "$SMACV2_ROOT" rev-parse HEAD)"
  echo "python=$($RUNTIME/bin/python -V 2>&1)"
} | tee "$RUN_DIR/runtime.txt"

nvidia-smi -L | tee "$RUN_DIR/nvidia-smi.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader \
  | tee -a "$RUN_DIR/nvidia-smi.txt"

cd "$CODE_ROOT"
"$RUNTIME/bin/python" -m experiments.clocked_async_mpg.run_two_clocks_hpc4_g0 \
  --harl-root "$HARL_ROOT" \
  --smacv2-root "$SMACV2_ROOT" \
  --output "$RUN_DIR/summary.json"

"$RUNTIME/bin/python" -m pip freeze --all | sort > "$RUN_DIR/pip-freeze.txt"
cp "$CODE_ROOT/docs/two_clocks_plot_contract.md" "$ROOT/plots/README.md"
echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RUN_DIR/runtime.txt"
sha256sum "$RUN_DIR"/* > "$RUN_DIR/SHA256SUMS"
