"""Seal the reviewed terminal record; no scientific outputs are recomputed."""
import ctypes
from ctypes import wintypes
import hashlib
import json
from pathlib import Path
import shutil
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
NATIVE=ROOT/'experiments/v3_observable_gate1h/results/formal_20261021'
ARCHIVE=ROOT/'ICML-2027/internal/v3_observable_gate1h_terminal_20261021'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def exited(pid):
    kernel=ctypes.windll.kernel32
    kernel.OpenProcess.restype=wintypes.HANDLE
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:return True
    value=wintypes.DWORD()
    try:
        if not kernel.GetExitCodeProcess(wintypes.HANDLE(handle),ctypes.byref(value)):raise ctypes.WinError()
        return value.value!=259
    finally:kernel.CloseHandle(wintypes.HANDLE(handle))

def run():
    assert sys.argv[1:]==['--scientific-cost-and-four-plots-reviewed']
    output=NATIVE/'independent_audited_terminal.json'
    assert not output.exists(),'Existing receipt preserved'
    assert all(exited(pid) for pid in (21644,25752,17284,1696))
    audit=json.loads((NATIVE/'independent_scientific_audit.json').read_text())
    runtime=json.loads((NATIVE/'independent_runtime_readback.json').read_text())
    prefix=json.loads((NATIVE/'independent_fixed_prefix_execution.json').read_text())
    assert audit['decision']=='MECHANISM_NOT_CONFIRMED_AT_REGISTERED_THRESHOLD'
    assert audit['science_rows']==audit['replay_rows']==294912 and audit['analyses']==2
    assert runtime['status']==prefix['status']=='PASS'
    assert prefix['all_32768_corrected_and_no_carryover_rows_exactly_equal']
    resource=dict(cap_seconds=3600,preparation_measured_process_tree_cpu_seconds=29.671875,
        allocated_preparation_auxiliary_seconds=60,exact_final_worker_cpu_seconds=1991.953125,
        exact_final_supervisor_cpu_seconds=73.78125,exact_final_scientific_runtime_audit_cpu_seconds=235.71875,
        exact_final_extra_prefix_audit_cpu_seconds=70.703125,allocated_terminal_auxiliary_seconds=45,
        auxiliary_allocations_are_not_measured_native_cpu=True,
        total_charged_cpu_seconds=2506.828125,remaining_seconds=1093.171875,
        total_charged_cpu_hours=2506.828125/3600,remaining_cpu_hours=1093.171875/3600,
        supervisor_terminal_snapshot_replaced_by_retained_handle_exact_final_cpu=True,
        independent_audit_exit_code=0,prefix_audit_exit_code=0,native_exit_codes=[0,0],
        no_budget_transfer_no_retry=True)
    assert sum(resource[k] for k in ('preparation_measured_process_tree_cpu_seconds','allocated_preparation_auxiliary_seconds',
        'exact_final_worker_cpu_seconds','exact_final_supervisor_cpu_seconds','exact_final_scientific_runtime_audit_cpu_seconds',
        'exact_final_extra_prefix_audit_cpu_seconds','allocated_terminal_auxiliary_seconds'))==resource['total_charged_cpu_seconds']
    sources=['workflow_complete.json','audited_terminal.json','worker_exit.json','independent_scientific_audit.json',
        'independent_runtime_readback.json','independent_fixed_prefix_execution.json']
    sources+=['analysis_a/'+name for name in ('summary.json','integrity.json','cell_results.csv','design_results.csv','primary_seed.csv','ledger_summary.csv')]
    sources+=['plots/'+name for name in ('risk_budget.png','risk_messages.png','risk_clock.png','risk_learner_budget.png')]
    copies={}
    for relative in sources:
        source=NATIVE/relative;target=ARCHIVE/source.name
        assert not target.exists(),'Preserve existing archive copy'
        shutil.copy2(source,target);assert sha(source)==sha(target)
        copies[relative]=sha(target)
    with (ARCHIVE/'resource.json').open('x',encoding='utf-8') as stream:json.dump(resource,stream,indent=2)
    report=ARCHIVE/'REPORT.txt';assert 'MECHANISM_NOT_CONFIRMED' in report.read_text(encoding='utf-8')
    receipt=dict(audit_complete=True,all_preserved_evidence_reviewed=True,terminal_decision_recorded=True,
        plots_visually_reviewed=True,terminal_decision=audit['decision']+'; no tuning, seed extension or scientific retry',
        native_root=str(NATIVE).replace('\\','/'),worker_pid=25752,supervisor_pid=21644,
        native_worker_and_supervisor_confirmed_exited=True,audit_pid=17284,audit_exit_code=0,
        prefix_audit_pid=1696,prefix_audit_exit_code=0,
        workflow_marker=str((NATIVE/'workflow_complete.json').relative_to(ROOT)).replace('\\','/'),
        workflow_marker_sha256=sha(NATIVE/'workflow_complete.json'),
        audit_report=str(report.relative_to(ROOT)).replace('\\','/'),audit_report_sha256=sha(report),
        science_rows=294912,replay_rows=294912,analyses=2,all_seven_analysis_bytes_equal=True,
        all_four_original_plots_inspected=True,total_charged_cpu_seconds=2506.828125,
        remaining_seconds_of_separate3600=1093.171875,manuscript_admission=False,no_scientific_retry=True,
        compact_evidence_sha256=copies,resource_sha256=sha(ARCHIVE/'resource.json'),
        independent_readback_not_external_human_review=True)
    for target in (output,ARCHIVE/'independent_audited_terminal.json'):
        with target.open('x',encoding='utf-8') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps(dict(status='AUDITED_TERMINAL_RECORDED',decision=audit['decision'],charge=2506.828125,
        report_sha256=sha(report),receipt_sha256=sha(output))))

if __name__=='__main__':run()
