#!/bin/bash
#SBATCH --job-name=exp014b-pilot
#SBATCH --partition=gpu-a30
#SBATCH --account=vincentlau
#SBATCH --qos=a30_qos
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=/scratch/jzhuangag/MARL-SDDE/worktrees/exp014b/logs/exp014b-%A_%a.out
#SBATCH --error=/scratch/jzhuangag/MARL-SDDE/worktrees/exp014b/logs/exp014b-%A_%a.err

set -euo pipefail
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export HF_HOME=/project/vincentlau/jzhuangag/hf_home
export HF_DATASETS_CACHE=/project/vincentlau/jzhuangag/hf_datasets_cache
export TORCH_HOME=/project/vincentlau/jzhuangag/torch_cache
export MPLCONFIGDIR=/scratch/jzhuangag/MARL-SDDE/worktrees/exp014b/.mplconfig

ROOT=/scratch/jzhuangag/MARL-SDDE/worktrees/exp014b
PYTHON=/project/vincentlau/jzhuangag/MARL-SDDE/envs/exp014a-py39/bin/python
RUN_ROOT=${ROOT}/experiments/nonlinear_markov_td/results/exp014b_pilot_20260731
SEED=$((20270821 + SLURM_ARRAY_TASK_ID))

mkdir -p "${MPLCONFIGDIR}" "${ROOT}/logs" "${RUN_ROOT}/seeds"
cd "${ROOT}/experiments/nonlinear_markov_td"
nvidia-smi
"${PYTHON}" run_hierarchical_controller_pilot.py \
  --device cuda \
  --seed "${SEED}" \
  --output-dir "${RUN_ROOT}/seeds/${SEED}"
