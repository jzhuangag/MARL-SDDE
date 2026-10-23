"""Fresh public metadata only; no manuscript upload and no bibliography edits."""
from datetime import datetime,timezone
import hashlib
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent;PAPER=HERE.parents[1]
spec=importlib.util.spec_from_file_location('prior49',PAPER/'internal/conditional_risk_20261021/audit_paper_references.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
adapter=prior.adapter;parse49=adapter.bib_entries
extra=PAPER/'quadratic_form_references_20261021.bib'
adapter.bib_entries=lambda text:parse49(text+'\n'+extra.read_text(encoding='utf-8'))
adapter.existing.RECORDS+=(adapter.existing.record('solomon1961quadratic',
    'On the Distribution of Quadratic Forms in Normal Variates',('Solomon',),1961,
    'https://digicoll.lib.berkeley.edu/record/112911'),)
dump_original=adapter.dump
def dump(path,value):
    if 'checked_on' in value:
        value=dict(value,checked_on=datetime.now(timezone.utc).date().isoformat(),
                   source_count=50,preserved_records_used=False,
                   new_auxiliary_bib_sha256=hashlib.sha256(extra.read_bytes()).hexdigest())
    return dump_original(path,value)
if __name__=='__main__':
    output=HERE/'fresh_citations';assert not output.exists()
    adapter.dump=dump;adapter.run(output,preserved_records=None)
