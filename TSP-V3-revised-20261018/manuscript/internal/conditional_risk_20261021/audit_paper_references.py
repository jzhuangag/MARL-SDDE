"""Fresh whole-paper49-reference lookup; no reuse of prior PASS records."""
from datetime import datetime,timezone
import hashlib
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
PAPER=HERE.parents[1]
spec=importlib.util.spec_from_file_location('current43',PAPER/'internal/proposition_literature_20261020/audit_references.py')
current=importlib.util.module_from_spec(spec);spec.loader.exec_module(current)
adapter=current.adapter;parse43=adapter.bib_entries
extra=PAPER/'conditional_references_20261021.bib'
adapter.bib_entries=lambda text:parse43(text+'\n'+extra.read_text(encoding='utf-8'))
record=adapter.existing.record
adapter.existing.RECORDS+=(
    record('moustakides2012joint','Joint Detection and Estimation: Optimum Tests and Applications',('Moustakides','Jajamovich','Tajer','Wang'),2012,'https://doi.org/10.1109/TIT.2012.2184260'),
    record('vasconcelos2020observation','Observation-Driven Scheduling for Remote Estimation of Two Gaussian Random Variables',('Vasconcelos','Mitra'),2020,'https://doi.org/10.1109/TCNS.2019.2900864'),
    record('reinhard2026minimax','Minimax Optimal Procedures for Joint Detection and Estimation',('Reinhard','Fauß','Zoubir'),2026,'https://arxiv.org/abs/2604.22740v1'),
    record('sion1958minimax','On general minimax theorems',('Sion',),1958,'https://doi.org/10.2140/pjm.1958.8.171'),
    record('harville1974contrasts','Bayesian inference for variance components using only error contrasts',('Harville',),1974,'https://doi.org/10.1093/biomet/61.2.383'),
    record('reinhard2024asymptotic','Asymptotically optimal procedures for sequential joint detection and estimation',('Reinhard','Fauß','Zoubir'),2024,'https://doi.org/10.1016/j.sigpro.2024.109410'),
)
original_dump=adapter.dump
started=datetime.now(timezone.utc).isoformat()
def dump(path,value):
    if 'checked_on' in value:
        value=dict(value,checked_on=datetime.now(timezone.utc).date().isoformat(),lookup_started_utc=started,preserved_records_used=False,source_count=49,additive_conditional_bib_sha256=hashlib.sha256(extra.read_bytes()).hexdigest())
    return original_dump(path,value)

if __name__=='__main__':
    adapter.dump=dump;adapter.run(HERE/'fresh_citations',preserved_records=None)
