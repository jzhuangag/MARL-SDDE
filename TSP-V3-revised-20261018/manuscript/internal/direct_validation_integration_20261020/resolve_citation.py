"""Fresh authoritative resolution of article-number semantics; no Bib edits."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

HERE=Path(__file__).resolve().parent
url="https://www.research.ed.ac.uk/en/publications/concentration-inequalities-for-markov-chains-by-marton-couplings-/"
with urllib.request.urlopen(url,timeout=40) as response:
    raw=response.read()
    http_status=response.status
with (HERE/"paulin_authoritative.html").open("xb") as file:
    file.write(raw)
text=raw.decode("utf-8")
checks={"title":"Concentration inequalities for Markov chains by Marton couplings and spectral methods" in text,
        "author":"Daniel Paulin" in text,"doi":"10.1214/EJP.v20-4039" in text,
        "article_number":"Article number" in text and ">79<" in text,
        "page_count":"Number of pages" in text and ">32<" in text,
        "volume":"Volume" in text and ">20<" in text,"year":"2015" in text}
assert all(checks.values()), checks
raw_report=json.loads((HERE/"fresh_citations/citation_verification.json").read_text(encoding="utf-8"))
assert len(raw_report["records"])==39
unresolved=[r for r in raw_report["records"] if r["status"]!="PASS"]
assert len(unresolved)==1 and unresolved[0]["key"]=="paulin2015concentration"
value=dict(checked_utc=datetime.now(timezone.utc).isoformat(),source=url,http_status=http_status,
           sha256=hashlib.sha256(raw).hexdigest(),checks=checks,raw_pass=38,resolved_article_number=79,
           volume=20,pages="1--32",year=2015,doi="10.1214/EJP.v20-4039",references=39,
           raw_report_unchanged=True,raw_resolver_failures_retained=True,bib_edited=False,
           conclusion="39 positive authoritative identities; sole raw bib_number conflict resolved as article-number representation, not a different work.",
           style=dict(definite_errors=[],judgments=["Article number in number field follows local existing representation.",
                       "Two arXiv journal strings are a local IEEE presentation choice; no unauthorized Bib rewrite."]))
with (HERE/"citation_resolution.json").open("x",encoding="utf-8") as file:
    json.dump(value,file,indent=2)
