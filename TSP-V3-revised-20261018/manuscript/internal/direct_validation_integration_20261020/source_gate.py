"""Check new paper patch against immutable before copies and audited tables."""
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from pypdf import PdfReader

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REPO=ROOT.parents[1]
read=lambda p:p.read_text(encoding="utf-8")
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
main=read(ROOT/"main.tex"); old=read(HERE/"before/main.tex")
supp=read(ROOT/"supplement.tex")
formal=lambda s:re.findall(r"\\begin\{(theorem|proposition|lemma|corollary|assumption)\}.*?\\end\{\1\}",s,re.S)
blocks=lambda s:re.findall(r"\\begin\{(?:theorem|proposition|lemma|corollary|assumption)\}.*?\\end\{(?:theorem|proposition|lemma|corollary|assumption)\}",s,re.S)
abstract=main.split("\\begin{abstract}")[1].split("\\end{abstract}")[0]
abstract_words=len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b",abstract))
keywords=main.split("\\begin{IEEEkeywords}")[1].split("\\end{IEEEkeywords}")[0].strip().rstrip(".").split(",")
old_transfer=old.split("\\subsection{Transfer across cooperative policy-learning environments}")[1].split("\\subsection{Ablations, complexity, and reproducibility}")[0].strip()
moved=read(ROOT/"transfer_details.tex").split("\n",1)[1].strip()
citekeys=lambda s:set(k.strip() for group in re.findall(r"\\cite\{([^}]+)\}",s) for k in group.split(","))
audit=json.loads(read(HERE/"fresh_citations/citation_verification.json"))
macros=list(csv.DictReader((REPO/"experiments/v3_risk_gate8h/results/formal_20261019/analysis_a/macro_results.csv").open(encoding="utf-8")))
table=supp.split("\\begin{tabular}{llrr}")[1].split("\\end{tabular}")[0]
lines=[line for line in table.splitlines() if "$" in line]
assert len(lines)==8
numeric=[]
for n,line in enumerate(lines):
    learner="running_mean" if n<4 else "averaged_delayed_affine"
    scale=(32,128,512,2048)[n%4]
    cells=re.findall(r"\$([^$]+)\$",line)
    assert len(cells)==2
    for regime,cell in zip(("high","low"),cells):
        values=[float(v) for v in re.findall(r"-?\d+(?:\.\d+)?",cell)]
        target=next(r for r in macros if r["policy"]=="risk_corrected_gate" and r["learner"]==learner and r["regime"]==regime and int(r["scale"])==scale)
        assert len(values)==3
        for val,key in zip(values,("mean_improvement","ci_low","ci_high")):
            assert abs(val-float(target[key]))<0.00001,(learner,regime,scale,key,val,target[key])
        numeric.append(dict(learner=learner,regime=regime,scale=scale,pass_=True))
checks={"main_formal_blocks_unchanged":blocks(main)==blocks(old),"formal_block_count":len(blocks(main)),
        "appendices_unchanged":sha(ROOT/"appendices.tex")==sha(HERE/"before/appendices.tex"),
        "both_bibliographies_unchanged":all(sha(ROOT/f)==sha(HERE/"before"/f) for f in ("references.bib","recent_work_20261019.bib")),
        "complete_old_transfer_moved_without_loss":old_transfer==moved,
        "abstract_words":abstract_words,"abstract_150_to_220":150<=abstract_words<=220,"five_keywords":len(keywords)==5,
        "citation_keys_match_fresh_audit":citekeys(main+read(ROOT/"appendices.tex"))==set(r["key"] for r in audit["records"]),
        "citation_keys_unchanged":citekeys(main)==citekeys(old),
        "no_new_qquad":all(read(ROOT/f).count("\\qquad")==read(HERE/"before"/f).count("\\qquad") for f in ("main.tex","supplement.tex")) and "\\qquad" not in moved,
        "preserved_math_qquad_count":main.count("\\qquad")+supp.count("\\qquad"),
        "new_prose_column_one":all(not line[:1].isspace() for f in ("main.tex","supplement.tex","transfer_details.tex") for line in read(ROOT/f).splitlines() if line.strip()),
        "main_pages":len(PdfReader(ROOT/"main.pdf").pages),"supplement_pages":len(PdfReader(ROOT/"supplement.pdf").pages),
        "original_v3_no_tracked_diff":not subprocess.check_output(["git","diff","--name-only","--","TSP-V3"],cwd=REPO,text=True).strip(),
        "original_v3_tree":subprocess.check_output(["git","rev-parse","HEAD:TSP-V3"],cwd=REPO,text=True).strip(),
        "all_16_macro_roundings_verified":numeric,
        "final_tex_sha256":sha(ROOT/"main.tex"),"final_supplement_sha256":sha(ROOT/"supplement.tex"),
        "pdf_sha256":{f:sha(ROOT/f) for f in ("main.pdf","supplement.pdf")},
        "log_checks":{f:not re.search(r"Overfull|undefined references|Reference .* undefined|multiply defined",read(ROOT/f)) for f in ("main.log","supplement.log")}}
checks["all_required_pass"]=all(checks[k] for k in ("main_formal_blocks_unchanged","appendices_unchanged","both_bibliographies_unchanged","complete_old_transfer_moved_without_loss","abstract_150_to_220","five_keywords","citation_keys_match_fresh_audit","citation_keys_unchanged","no_new_qquad","new_prose_column_one","original_v3_no_tracked_diff")) and checks["main_pages"]==13 and all(checks["log_checks"].values())
with (HERE/(sys.argv[1] if len(sys.argv)>1 else "source_gate_corrected.json")).open("x",encoding="utf-8") as file:json.dump(checks,file,indent=2)
print(json.dumps(checks,indent=2))
assert checks["all_required_pass"]
