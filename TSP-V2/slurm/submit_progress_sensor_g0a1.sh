#!/bin/bash

set -euo pipefail

ROOT=/scratch/jzhuangag/MARL-SDDE-TSP-V2-PROGRESS-G0A1
CODE="$ROOT/code/MARL-SDDE"
SBATCH="$CODE/TSP-V2/slurm/progress_sensor_g0a1_a30.sbatch"

if [ -e "$ROOT/artifacts" ] && find "$ROOT/artifacts" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to submit over existing artifacts" >&2
  exit 2
fi

mkdir -p "$ROOT/logs" "$ROOT/artifacts" "$ROOT/tmp"
test -d "$ROOT/logs"
test -d "$ROOT/artifacts"
test -d "$ROOT/tmp"
test -f "$SBATCH"
test "$(git -C "$CODE" status --porcelain | wc -l)" -eq 0

sbatch "$SBATCH"

