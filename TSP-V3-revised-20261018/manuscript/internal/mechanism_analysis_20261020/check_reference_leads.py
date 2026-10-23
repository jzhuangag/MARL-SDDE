"""Bibliographic leads for the NEW mechanism argument, no manuscript/Bib edits."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

START=time.process_time();HERE=Path(__file__).resolve().parent
records=[]
for doi in ("10.1214/ss/1177009939","10.1214/aoms/1177729032","10.1111/j.2517-6161.1961.tb00408.x"):
    sources={}
    for name,url in (("Crossref","https://api.crossref.org/works/"+urllib.parse.quote(doi,safe="")),
                     ("OpenAlex","https://api.openalex.org/works/https://doi.org/"+doi)):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"V3 mechanism bibliographic verification"}),timeout=25) as response:
                raw=response.read();status=response.status
            value=json.loads(raw)
            sources[name]=dict(http_status=status,sha256=hashlib.sha256(raw).hexdigest(),metadata=value)
        except (urllib.error.URLError,TimeoutError,json.JSONDecodeError) as error:
            sources[name]=dict(status="UNAVAILABLE_NOT_PASS",error=str(error))
    records.append(dict(doi=doi,resolvers=sources,manuscript_admitted=False))
report=dict(checked_utc=datetime.now(timezone.utc).isoformat(),records=records,
    pmlr=dict(url="https://proceedings.mlr.press/v51/russo16.html",title="Controlling Bias in Adaptive Data Analysis Using Information Theory",
              authors=["Daniel Russo","James Zou"],year=2016,pages="1232--1240",publisher_series_volume=51,
              pdf_footer_volume=41,metadata_conflict_retained=True,manuscript_admitted=False,
              note="Fresh official proceedings page and PDF opened. Resolve the PDF footer's conflicting series-volume representation before introducing a new Bib entry."),
    scope="These are research leads and novelty cautions, not a new final bibliography audit or exhaustive full-text priority clearance.",
    observed_analysis_cpu_seconds=time.process_time()-START)
with (HERE/"new_reference_leads.json").open("x",encoding="utf-8") as file:json.dump(report,file,indent=2)
print(json.dumps(dict(records=len(records),cpu_seconds=report["observed_analysis_cpu_seconds"],
    statuses=[dict(doi=r["doi"],resolvers={k:v.get("http_status",v.get("status")) for k,v in r["resolvers"].items()}) for r in records])))
