#!/usr/bin/env bash
set -euo pipefail

# Operational continuation for the frozen TSP-MARL-MPE-QGRID-AUDIT-001
# design.  This script never retries a failed scientific cell.  It submits the
# next immutable eight-cell batch only after every task in the preceding batch
# has completed with exit code 0:0.

ROOT=/scratch/jzhuangag/MARL-SDDE-TSP-MPE-QGRID-AUDIT-001
CODE="$ROOT/code/MARL-SDDE"
SBATCH_FILE="$CODE/TSP/slurm/marl_mpe_qgrid_audit_a30.sbatch"
LOG="$ROOT/logs/mpe-qgrid-continuation.log"
ARTIFACT_SET=recovery1
POLL_SECONDS=60

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CURRENT_ARRAY_JOB_ID" >&2
  exit 2
fi

current_job="$1"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG"
}

wait_for_success() {
  local job_id="$1"
  while squeue -h -j "$job_id" | grep -q .; do
    sleep "$POLL_SECONDS"
  done

  local audit
  audit=$(sacct -X -n -P -j "$job_id" -o JobID,State,ExitCode | \
    awk -F'|' -v prefix="${job_id}_" '$1 ~ ("^" prefix "[0-9]+$") {print}')
  local count
  count=$(printf '%s\n' "$audit" | awk 'NF {n++} END {print n+0}')
  if [[ "$count" -ne 8 ]]; then
    log "STOP job=$job_id reason=expected-8-array-tasks observed=$count"
    printf '%s\n' "$audit" | tee -a "$LOG"
    return 1
  fi
  if printf '%s\n' "$audit" | awk -F'|' '$2 != "COMPLETED" || $3 != "0:0" {bad=1} END {exit bad ? 0 : 1}'; then
    log "STOP job=$job_id reason=nonzero-or-incomplete-task"
    printf '%s\n' "$audit" | tee -a "$LOG"
    return 1
  fi
  log "VERIFIED job=$job_id tasks=8 state=COMPLETED exit=0:0"
}

submit_batch() {
  local offset="$1"
  sbatch --parsable \
    --export="ALL,OFFSET=$offset,ARTIFACT_SET=$ARTIFACT_SET" \
    --output="$ROOT/logs/mpe-r1-%A_%a.out" \
    --error="$ROOT/logs/mpe-r1-%A_%a.err" \
    "$SBATCH_FILE"
}

log "START current_job=$current_job artifact_set=$ARTIFACT_SET"
for offset in 8 16 24; do
  wait_for_success "$current_job"
  current_job=$(submit_batch "$offset")
  log "SUBMITTED job=$current_job offset=$offset"
done
wait_for_success "$current_job"
log "COMPLETE all=32-new-runs artifact_set=$ARTIFACT_SET"
