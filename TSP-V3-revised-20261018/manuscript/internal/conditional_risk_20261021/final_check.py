"""Verify research-package integrity and absence of manuscript/science mutation."""
from pathlib import Path
import hashlib
import json
import subprocess
import time
START=time.process_time()
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
OUTPUT=HERE/'release_check.json'
if OUTPUT.exists():raise RuntimeError('Preserve existing release check')
base=json.loads((HERE/'intent.json').read_text(encoding='utf-8'))['baseline_commit']
paper='TSP-V3-revised-20261018/manuscript/'
paths=[paper+name for name in ('main.tex','supplement.tex','main.pdf','supplement.pdf',
                              'mechanism_details.tex','appendices.tex')]
paths+=subprocess.check_output(['git','ls-files',paper+'*.bib'],cwd=REPO,text=True).splitlines()
paths += ['experiments/v3_risk_gate8h/'+name for name in (
    'core.py','protocol.json','execution_spec.json','launch.py',
    'registration_v2/lock.json','registration_v2/admission.json')]
preserved=[]
for path in paths:
    raw=(REPO/path).read_bytes()
    old=subprocess.check_output(['git','show',base+':'+path],cwd=REPO)
    assert raw==old,path
    preserved.append(dict(path=path,sha256=hashlib.sha256(raw).hexdigest(),baseline_identical=True))
proof=json.loads((HERE/'proof_checks.json').read_text(encoding='utf-8'))
robust=json.loads((HERE/'robust_boundary_checks.json').read_text(encoding='utf-8'))
minimax=json.loads((HERE/'minimax_envelope_checks.json').read_text(encoding='utf-8'))
assert proof['status']==robust['status']==minimax['status']=='PASS'
assert proof['passed']==46 and robust['cases']==7 and minimax['cases']==3
proposal=json.loads((HERE/'proposed_validation.json').read_text(encoding='utf-8'))
assert 9*2*16*4*2*2*64==proposal['counts']['formal_policy_trajectories']==294912
assert proposal['counts']['semantic_replay_policy_trajectories']==294912
assert proposal['resource_request']['separate_total_cpu_seconds']==7200
text=(HERE/'theory.tex').read_text(encoding='utf-8')
assert not any(line[:1].isspace() for line in text.splitlines() if line)
assert '\\qquad' not in text
hashes={}
script_cpu=0.
for path in sorted(HERE.iterdir()):
    if path.is_file() and path.name not in ('release_check.json',) and not path.name.endswith('.source.html'):
        raw=path.read_bytes();hashes[path.name]=hashlib.sha256(raw).hexdigest()
        if path.suffix=='.json':
            data=json.loads(raw)
            script_cpu+=data.get('cpu_seconds',0) or 0
result=dict(status='PASS_RESEARCH_PACKAGE_NOT_SUBMISSION_READINESS',
    baseline_commit=base,manuscript_and_scientific_inputs=preserved,
    mathematical_checks=dict(first_fixture_checks=46,robust_boundary_cases=7,
        exhaustive_symmetric_cases=3,selectors_per_exhaustive_case=729,
        random_number_calls=0,new_scientific_samples=0),
    proposal_status=proposal['status'],new_science_launched=False,
    hashes=hashes,measured_script_cpu_seconds_excluding_this_check=script_cpu,
    conservative_unmetered_auxiliary_charge_seconds=120,
    auxiliary_note='Allocated conservative ordinary-research charge, not measured native CPU; includes failed import, shell/network/Git overhead. No completed study budget was changed.',
    this_check_cpu_seconds=time.process_time()-START,
    review='Single-session proof/source/adversarial checks; no independent expert review claimed',
    manuscript_pdf_release='UNCHANGED; no new manuscript/PDF deliverable is asserted')
result['ordinary_research_charge_seconds']=script_cpu+120+result['this_check_cpu_seconds']
with OUTPUT.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
print(json.dumps(dict(status=result['status'],preserved_files=len(preserved),
    ordinary_research_charge_seconds=result['ordinary_research_charge_seconds'])))
