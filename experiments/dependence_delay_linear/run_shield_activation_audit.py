"""Emit frozen analytic scan to stdout; no scientific result-directory writes."""
import itertools
import json
from pathlib import Path
from experiments.dependence_delay_linear.shield_activation_audit import activation_path


def scan():
    root=Path(__file__).resolve().parents[2]
    plan=json.loads((root/'docs/shield_activation_scan_plan.json').read_text())
    rows=[]
    for m,eta,lam,scale,schedule in itertools.product(plan['block_lengths'],plan['gains'],plan['temporal_correlations'],plan['target_scales'],plan['schedules']):
        targets=[scale if schedule=='constant' or k<plan['blocks']//2 else -scale for k in range(plan['blocks'])]
        path=activation_path(targets,m,eta,lam,plan['variance'],plan['initial'],plan['agents'],plan['delta'])
        rows.append(dict(m=m,eta=eta,lam=lam,scale=scale,schedule=schedule,
                         mean_probability=sum(p['probability'] for p in path)/len(path),path=path))
    return dict(id=plan['id'],rows=rows)


if __name__=='__main__':
    print(json.dumps(scan(),sort_keys=True))
