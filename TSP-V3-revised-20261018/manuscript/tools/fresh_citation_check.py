"""Fresh public-metadata lookup for the isolated revision, with no cached passes."""
from datetime import datetime, timezone
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
SOURCE = ROOT/'TSP-V3/manuscript/tools/final_citation_audit_20261016.py'
spec = importlib.util.spec_from_file_location('v3_bibliographic_resolvers', SOURCE)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
adapter.HERE = HERE
original_dump = adapter.dump


def dated_dump(path, value):
    if 'checked_on' in value:
        value = dict(value, checked_on=datetime.now(timezone.utc).date().isoformat(),
                     lookup_started_utc=started, preserved_records_used=False)
    return original_dump(path, value)


if __name__ == '__main__':
    started = datetime.now(timezone.utc).isoformat()
    adapter.dump = dated_dump
    adapter.run(HERE/'internal/fresh_citation_20261018', preserved_records=None)
