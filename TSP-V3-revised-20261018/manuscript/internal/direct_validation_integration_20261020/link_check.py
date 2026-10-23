import hashlib
import json
import sys
from pathlib import Path
from pypdf import PdfReader

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
main=PdfReader(ROOT/"main.pdf")
supp=PdfReader(ROOT/"supplement.pdf")
keys=("dalfabbro2024dasa","perazzone2025lyapunov","yu2025sdde","saad2023covariance","li2026parameterfree","li2026shrinkingtube")
actions=[str(a.get_object().get("/A","")) for p in supp.pages for a in p.get("/Annots",[])]
results={key:dict(main_destination_exists="cite."+key in main.named_destinations,
                  external_link_exists=any("main.pdf" in a and "cite."+key in a for a in actions)) for key in keys}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prefix=sys.argv[1] if len(sys.argv)>1 else "supplinks"
pixels={i:sha(HERE/f"{prefix}-{i:02d}.png")==sha(HERE/"hyperlink_before"/f"supplement-{i:02d}.png") for i in range(1,11)}
value=dict(citation_links=results,all_ten_rendered_pages_byte_identical=pixels,
           final_supplement_pdf_sha256=sha(ROOT/"supplement.pdf"),
           log_has_missing_destinations="has been referenced but" in (ROOT/"supplement.log").read_text(encoding="utf-8"))
with (HERE/(prefix+"_check.json")).open("x",encoding="utf-8") as file:json.dump(value,file,indent=2)
print(json.dumps(value))
assert all(all(r.values()) for r in results.values()) and all(pixels.values()) and not value["log_has_missing_destinations"]
