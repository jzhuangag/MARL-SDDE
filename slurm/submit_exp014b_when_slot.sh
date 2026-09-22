#!/bin/bash
# Submit EXP-014B pilot seeds as Slurm quota becomes available.

set -u

ROOT=/scratch/jzhuangag/MARL-SDDE/worktrees/exp014b
LOG_DIR=${ROOT}/logs
LEDGER=${LOG_DIR}/exp014b-submitted.tsv
LOCK=${LOG_DIR}/exp014b-submit.lock
SBATCH_FILE=${ROOT}/slurm/exp014b_pilot_a30.sbatch

mkdir -p "${LOG_DIR}"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "another EXP-014B submitter already holds ${LOCK}" >&2
  exit 1
fi

if [[ ! -f "${LEDGER}" ]]; then
  printf 'array_index\tseed\tjob_id\tsubmitted_at\n' >"${LEDGER}"
fi

for array_index in $(seq 0 7); do
  seed=$((20270821 + array_index))
  if awk -F '\t' -v target_index="${array_index}" \
      'NR > 1 && $1 == target_index { found = 1 } END { exit !found }' \
      "${LEDGER}"; then
    continue
  fi
  while true; do
    submitted_at=$(date --iso-8601=seconds)
    if job_id=$(sbatch --parsable \
        --array="${array_index}-${array_index}" "${SBATCH_FILE}" 2>/dev/null); then
      printf '%s\t%s\t%s\t%s\n' \
        "${array_index}" "${seed}" "${job_id}" "${submitted_at}" >>"${LEDGER}"
      break
    fi
    sleep 60
  done
done
