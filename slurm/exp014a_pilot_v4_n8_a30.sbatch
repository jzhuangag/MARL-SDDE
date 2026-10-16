#!/bin/bash
#SBATCH --job-name=exp014a-v4-n8
#SBATCH --partition=gpu-a30
#SBATCH --account=vincentlau
#SBATCH --qos=a30_qos
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/jzhuangag/MARL-SDDE/logs/exp014a-v4-n8-%j.out
#SBATCH --error=/scratch/jzhuangag/MARL-SDDE/logs/exp014a-v4-n8-%j.err

set -euo pipefail
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export HF_HOME=/project/vincentlau/jzhuangag/hf_home
export HF_DATASETS_CACHE=/project/vincentlau/jzhuangag/hf_datasets_cache
export TORCH_HOME=/project/vincentlau/jzhuangag/torch_cache
export MPLCONFIGDIR=/scratch/jzhuangag/MARL-SDDE/.mplconfig

ROOT=/scratch/jzhuangag/MARL-SDDE
PYTHON=/project/vincentlau/jzhuangag/MARL-SDDE/envs/exp014a-py39/bin/python
OUTPUT=${ROOT}/experiments/nonlinear_markov_td/results/state_risk_controller_smoke_v4_n8

mkdir -p "${MPLCONFIGDIR}" "${OUTPUT}"
cd "${ROOT}/experiments/nonlinear_markov_td"
nvidia-smi
"${PYTHON}" run_state_risk_controller_smoke.py \
  --device cuda \
  --num-seeds 8 \
  --base-seed 20270801 \
  --message-budget 32000 \
  --updates-per-block 32 \
  --output-dir "${OUTPUT}"
