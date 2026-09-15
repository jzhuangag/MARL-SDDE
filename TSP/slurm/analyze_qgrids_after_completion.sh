#!/usr/bin/env bash
set -euo pipefail

# Reproduce the two frozen q-grid analyses after their fail-closed execution
# ledgers report success.  Scientific gates may pass or fail; this script never
# changes a threshold and never launches a controller experiment.

MPE_ROOT=/scratch/jzhuangag/MARL-SDDE-TSP-MPE-QGRID-AUDIT-001
MPE_BASE=/scratch/jzhuangag/MARL-SDDE-TSP-MARL-CONF-001
SMAC_ROOT=/scratch/jzhuangag/MARL-SDDE-TSP-SMACV2-QGRID-DEV-001
SMAC_BASE=/scratch/jzhuangag/MARL-SDDE-TSP-SMACV2-DEV-001/artifacts/headroom
MPE_LOG="$MPE_ROOT/logs/mpe-qgrid-continuation.log"
SMAC_LOG="$SMAC_ROOT/logs/smac-qgrid-continuation.log"
STATUS_LOG="$SMAC_ROOT/logs/qgrid-analysis-continuation.log"
POLL_SECONDS=60
export SOURCE_DATE_EPOCH=0
export TZ=UTC

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$STATUS_LOG"
}

wait_for_marker() {
  local file="$1"
  local marker="$2"
  while ! grep -q "$marker" "$file" 2>/dev/null; do
    sleep "$POLL_SECONDS"
  done
}

run_mpe_analysis() {
  local code="$MPE_ROOT/code/MARL-SDDE"
  local python="$MPE_BASE/venv/bin/python"
  local out_a="$MPE_ROOT/analysis"
  local out_b="$MPE_ROOT/analysis_replay"
  [[ ! -e "$out_a" && ! -e "$out_b" ]] || {
    log "STOP experiment=MPE reason=analysis-output-already-exists"
    return 1
  }
  mkdir -p "$out_a" "$out_b"
  for out in "$out_a" "$out_b"; do
    PYTHONPATH="$code/TSP/experiments" "$python" \
      "$code/TSP/experiments/analyze_mappo_mpe_qgrid_audit.py" \
      --existing-root "$MPE_BASE" \
      --extension-root "$MPE_ROOT" \
      --summary-csv "$out/curves.csv" \
      --figure "$out/return_curves.pdf" \
      --gate-json "$out/gates.json" \
      --replay-identical > "$out/stdout.txt"
  done
  cmp "$out_a/curves.csv" "$out_b/curves.csv"
  cmp "$out_a/return_curves.pdf" "$out_b/return_curves.pdf"
  cmp "$out_a/gates.json" "$out_b/gates.json"
  (cd "$out_a" && sha256sum curves.csv return_curves.pdf gates.json > SHA256SUMS)
  log "ANALYZED experiment=TSP-MARL-MPE-QGRID-AUDIT-001 replay=byte-identical"
}

run_smac_analysis() {
  local code="$SMAC_ROOT/code/MARL-SDDE"
  local python=/scratch/jzhuangag/MARL-SDDE-TwoClocks-20260902/envs/runtime-py39/bin/python
  local out_a="$SMAC_ROOT/analysis"
  local out_b="$SMAC_ROOT/analysis_replay"
  [[ ! -e "$out_a" && ! -e "$out_b" ]] || {
    log "STOP experiment=SMACv2 reason=analysis-output-already-exists"
    return 1
  }
  mkdir -p "$out_a" "$out_b"
  for out in "$out_a" "$out_b"; do
    PYTHONPATH="$code/TSP/experiments" "$python" \
      "$code/TSP/experiments/analyze_mappo_smacv2_qgrid.py" \
      --existing-root "$SMAC_BASE" \
      --extension-root "$SMAC_ROOT" \
      --summary-csv "$out/curves.csv" \
      --figure "$out/return_curves.pdf" \
      --gate-json "$out/gates.json" \
      --replay-identical > "$out/stdout.txt"
  done
  cmp "$out_a/curves.csv" "$out_b/curves.csv"
  cmp "$out_a/return_curves.pdf" "$out_b/return_curves.pdf"
  cmp "$out_a/gates.json" "$out_b/gates.json"
  (cd "$out_a" && sha256sum curves.csv return_curves.pdf gates.json > SHA256SUMS)
  log "ANALYZED experiment=TSP-MARL-SMACV2-QGRID-DEV-001 replay=byte-identical"
}

log "START"
wait_for_marker "$MPE_LOG" 'COMPLETE all=32-new-runs'
run_mpe_analysis
wait_for_marker "$SMAC_LOG" 'COMPLETE job='
run_smac_analysis
log "COMPLETE both=qgrid-analyses"
