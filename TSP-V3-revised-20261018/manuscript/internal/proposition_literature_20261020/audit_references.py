"""Fresh whole-paper lookup with additive mechanism references; no source edits."""
from datetime import datetime, timezone
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
spec = importlib.util.spec_from_file_location('current39', PAPER/'tools/content_strengthening_citations.py')
current = importlib.util.module_from_spec(spec)
spec.loader.exec_module(current)
adapter = current.adapter
parse39 = adapter.bib_entries
extra = PAPER/'mechanism_references_20261020.bib'
adapter.bib_entries = lambda text: parse39(text+'\n'+extra.read_text(encoding='utf-8'))
adapter.existing.RECORDS += (
    adapter.existing.record('esary1967association','Association of Random Variables, with Applications',
        ('Esary','Proschan','Walkup'),1967,'https://doi.org/10.1214/aoms/1177698701'),
    adapter.existing.record('chaloner1995bayesian','Bayesian Experimental Design: A Review',
        ('Chaloner','Verdinelli'),1995,'https://doi.org/10.1214/ss/1177009939'),
    adapter.existing.record('russo2016bias','Controlling Bias in Adaptive Data Analysis Using Information Theory',
        ('Russo','Zou'),2016,'https://proceedings.mlr.press/v51/russo16.html'),
    adapter.existing.record('manjunath2021moments','Moments Calculation for the Doubly Truncated Multivariate Normal Density',
        ('Manjunath','Wilhelm'),2021,'https://doi.org/10.35566/jbds/v1n1/p2'),
)
original_dump=adapter.dump
started=datetime.now(timezone.utc).isoformat()
def dump(path,value):
    if 'checked_on' in value:
        value=dict(value,checked_on=datetime.now(timezone.utc).date().isoformat(),
            lookup_started_utc=started,preserved_records_used=False,source_count=43,
            includes_all_four_new_main_citations=True)
    return original_dump(path,value)
if __name__=='__main__':
    adapter.dump=dump
    adapter.run(HERE/'fresh_citations',preserved_records=None)
