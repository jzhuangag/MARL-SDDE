"""One-shot pre-data registration read-back; never launches a native worker."""
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name] = "1"
import json
import time
from guard import HERE,LOCK,ADMISSION,RESULT,sha,dump,verify_registration,predecessor_ready


if __name__=="__main__":
    if RESULT.exists():
        raise RuntimeError("Readiness check must precede scientific data")
    lock,admission,git_cpu = verify_registration()
    ready,reason = predecessor_ready()
    previous_parent = sum(lock["preparation_parent_snapshots_seconds"])+lock["freeze_parent_cpu_snapshot_seconds"]
    aux_observed = previous_parent+time.process_time()+git_cpu
    if aux_observed>=60:
        raise AssertionError("Auxiliary preparation exceeded upfront charge; do not launch")
    charged = lock["earlier_charge_seconds"]
    report = dict(status="REGISTRATION_VERIFIED_WAITING_AUDITED_140H_HANDOFF" if not ready else "REGISTRATION_VERIFIED_READY",
                  latest_test_count=lock["latest_test_count"],test_attempts=lock["all_test_attempts"],
                  all_source_runtime_and_pushed_registration_bytes_match=True,
                  lock_sha256=sha(LOCK),admission_sha256=sha(ADMISSION),
                  formal_science_rows=0,formal_replay_rows=0,analysis_copies=0,
                  source_and_metadata_hashes=len(lock["source_sha256"]),
                  predecessor_ready=ready,predecessor_reason=reason,native_launched=False,
                  preparation_measured_test_cpu_seconds=lock["preparation_measured_cpu_seconds"],
                  auxiliary_observed_snapshots_cpu_seconds=aux_observed,
                  auxiliary_conservative_charge_seconds=60.,cumulative_charged_cpu_seconds=charged,
                  remaining_cpu_seconds=28800-charged,cap_seconds=28800,
                  positive_negative_partial_failed_evidence_preserved=True)
    dump(HERE/"preparation/readiness.json",report)
    print(json.dumps(report,sort_keys=True))
