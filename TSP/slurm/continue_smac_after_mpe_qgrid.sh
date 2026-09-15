#!/usr/bin/env bash
set -euo pipefail

# Submit the independently frozen SMACv2 q-grid only after the MPE q-grid
# continuation has verified all of its batches.  This is an operational
# dependency, not an outcome-dependent benchmark choice.

MPE_ROOT=/scratch/jzhuangag/MARL-SDDE-TSP-MPE-QGRID-AUDIT-001
SMAC_ROOT=/scratch/jzhuangag/MARL-SDDE-TSP-SMACV2-QGRID-DEV-001
SBATCH_FILE="$SMAC_ROOT/code/MARL-SDDE/TSP/slurm/marl_smacv2_qgrid_extension_a30.sbatch"
MPE_LOG="$MPE_ROOT/logs/mpe-qgrid-continuation.log"
LOG="$SMAC_ROOT/logs/smac-qgrid-continuation.log"
POLL_SECONDS=60

if [[ $# -ne 1 ]]; then
  echo "usage: $0 MPE_CONTINUATION_PID" >&2
  exit 2
fi

mpe_pid="$1"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG"
}

while ! grep -q 'COMPLETE all=32-new-runs' "$MPE_LOG" 2>/dev/null; do
  if ! kill -0 "$mpe_pid" 2>/dev/null; then
    log "STOP reason=mpe-continuation-ended-without-success pid=$mpe_pid"
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

job_id=$(sbatch --parsable \
  --output="$SMAC_ROOT/logs/smac-qgrid-%A_%a.out" \
  --error="$SMAC_ROOT/logs/smac-qgrid-%A_%a.err" \
  "$SBATCH_FILE")
log "SUBMITTED job=$job_id cells=8"

while squeue -h -j "$job_id" | grep -q .; do
  sleep "$POLL_SECONDS"
done

audit=$(sacct -X -n -P -j "$job_id" -o JobID,State,ExitCode | \
  awk -F'|' -v prefix="${job_id}_" '$1 ~ ("^" prefix "[0-9]+$") {print}')
count=$(printf '%s\n' "$audit" | awk 'NF {n++} END {print n+0}')
if [[ "$count" -ne 8 ]]; then
  log "STOP job=$job_id reason=expected-8-array-tasks observed=$count"
  printf '%s\n' "$audit" | tee -a "$LOG"
  exit 1
fi
if printf '%s\n' "$audit" | awk -F'|' '$2 != "COMPLETED" || $3 != "0:0" {bad=1} END {exit bad ? 0 : 1}'; then
  log "STOP job=$job_id reason=nonzero-or-incomplete-task"
  printf '%s\n' "$audit" | tee -a "$LOG"
  exit 1
fi
log "COMPLETE job=$job_id tasks=8 state=COMPLETED exit=0:0"
