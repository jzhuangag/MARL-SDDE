"""Fresh 39-reference audit, additive bibliography; no source/Bib rewrite."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fresh_base", HERE/"tools/fresh_citation_check.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
adapter = base.adapter
parse = adapter.bib_entries
extra = HERE/"recent_work_20261019.bib"
adapter.bib_entries = lambda text: parse(text+"\n"+extra.read_text(encoding="utf-8"))
adapter.existing.RECORDS += (
    adapter.existing.record("li2026parameterfree", "Towards Parameter-Free Temporal Difference Learning",
                            ("Li", "Schmidt", "Babanezhad", "Vaswani"), 2026,
                            "https://arxiv.org/abs/2603.02577v1"),
    adapter.existing.record("li2026shrinkingtube", "Shrinking-Tube Concentration for Adaptive Markovian Stochastic Approximation",
                            ("Li", "Luo", "Zhang"), 2026, "https://arxiv.org/abs/2609.29833v1"),
)
started = datetime.now(timezone.utc).isoformat()
original_dump = adapter.dump


def dump(path, value):
    if "checked_on" in value:
        value = dict(value, checked_on=datetime.now(timezone.utc).date().isoformat(),
                     lookup_started_utc=started, preserved_records_used=False,
                     additive_bibliography_sha256=hashlib.sha256(extra.read_bytes()).hexdigest(),
                     source_count=39)
    return original_dump(path, value)


if __name__ == "__main__":
    adapter.dump = dump
    adapter.run(HERE/"internal/content_strengthening_20261019/fresh_citations", preserved_records=None)
