"""New uncached run; retain raw resolver failures and do not change BibTeX."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = HERE.parents[1] / "tools/content_strengthening_citations.py"
spec = importlib.util.spec_from_file_location("new_citation_audit", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.adapter.dump = module.dump
module.adapter.run(HERE / "fresh_citations", preserved_records=None)
