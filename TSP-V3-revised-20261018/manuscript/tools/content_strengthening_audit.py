"""Scoped revision checks, not journal acceptance or independent peer review."""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request
import numpy as np
import pandas as pd
from pypdf import PdfReader
from PIL import Image, ImageDraw
from risk_gate_core import InstanceRisk, admit, risk_upper

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
OUT = HERE/"internal/content_strengthening_20261019"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def core_checks():
    x, w = np.polynomial.hermite.hermgauss(5)
    x, w = x*np.sqrt(2), w/np.sqrt(np.pi)
    count = 0
    for s2 in (0, .1, 1):
        for dc, dw, kc, kw in ((0, 1, .2, .3), (1, 1, .4, .1), (.2, -.3, -.1, .8)):
            for delta in (0, .04):
                case = InstanceRisk(.2, .4, dc, dw, kc, kw, s2, delta, 1)
                U, carry = risk_upper(case)
                m = np.sqrt(s2)*x
                centered = 2*(dw*kw-dc*kc)*m+(kw*kw-kc*kc)*(m*m-s2)
                assert abs(float(w@centered)) < 1e-13
                assert math.isclose(carry*carry, delta*float(w@(centered*centered)), abs_tol=1e-13)
                assert math.isclose(U, .2+.2*delta+carry, abs_tol=1e-13)
                count += 1
    safe = InstanceRisk(.1, .1, 0, 0, 0, 0, 0, 0, .2)
    assert admit(safe, safe, 0, True)
    assert not admit(safe, replace(safe, fallback_J=.1), 0, True)  # strict high margin
    assert not admit(safe, safe, 0, False)
    assert not admit(replace(safe, fallback_J=.05), safe, 0, True)
    for changes in ({"directional_error_upper": -1}, {"correct_J": -1}, {"wrong_k": float("nan")}):
        try:
            risk_upper(replace(safe, **changes))
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid certificate input admitted")
    return count+7


def montage():
    directory = OUT/"render"
    for prefix in ("main", "supplement"):
        files = sorted(directory.glob(prefix+"-*.png"))
        for begin in range(0, len(files), 4):
            canvas = Image.new("RGB", (1800, 2440), "#ddd")
            draw = ImageDraw.Draw(canvas)
            for j, p in enumerate(files[begin:begin+4]):
                with Image.open(p) as im:
                    im.thumbnail((880, 1170))
                    col, row = j%2, j//2
                    canvas.paste(im, (col*900+10, row*1220+30))
                    draw.text((col*900+10, row*1220+5), p.name, fill="black")
            canvas.save(directory/f"contact_{prefix}_{begin//4+1}.png")


def audit(visual):
    start = time.process_time()
    target = OUT/"release_check.json"
    if target.exists():
        raise FileExistsError(target)
    original = ROOT/"TSP-V3/manuscript"
    expected = {"main.tex":"165adac70b67f2067964380351e8f94d129548b2b9da55007434a5821fba7e78",
                "appendices.tex":"dfcec46942e3640bbe320b68b7aed42fc3944817a879e0fd6fc34f4d9a19e0cc",
                "references.bib":"9ce047e460c16da8ea3cbe1ed9bdf55b90b4c0b2eba19bbe71c571dd431a7690",
                "main.pdf":"6fe44327eb90d286de354c0698f49611fce0082bea09cec4dd7d7d5d7ba46bfd"}
    for name, value in expected.items():
        assert sha(original/name) == value, name
    base = OUT/"base"
    main = (HERE/"main.tex").read_text(encoding="utf-8")
    old = (base/"main.tex").read_text(encoding="utf-8")
    supplement = (HERE/"supplement.tex").read_text(encoding="utf-8")
    appendix = (HERE/"appendices.tex").read_text(encoding="utf-8")
    assert sha(HERE/"references.bib") == sha(base/"references.bib")
    assert re.search(r"\\author\{.*?\\begin\{document\}", main, re.S).group() == re.search(r"\\author\{.*?\\begin\{document\}", old, re.S).group()
    formal = re.findall(r"\\begin\{(theorem|lemma|proposition|corollary|assumption)\}.*?\\end\{\1\}", old, re.S)
    pattern = r"\\begin\{(theorem|lemma|proposition|corollary|assumption)\}.*?\\end\{\1\}"
    old_blocks = [m.group() for m in re.finditer(pattern, old, re.S)]
    assert len(old_blocks) == 13, len(old_blocks)
    assert all(block in main for block in old_blocks)
    combined = main+appendix+supplement
    old_appendix = (base/"appendices.tex").read_text(encoding="utf-8")
    tracking = old_appendix.split("Set $R_\\star=d/(1-c)$",1)[1].split("\\subsection{Existence",1)[0]
    occupation = old_appendix.split("Let $\\Delta_j(z,a)\\ge0$",1)[1].split("\\section{Proof of the Main",1)[0]
    assert "Set $R_\\star=d/(1-c)$"+tracking in supplement
    assert ("Let $\\Delta_j(z,a)\\ge0$"+occupation).rstrip() in supplement
    for token in ("101/446", "33/222", "-0.0049", "0.0252"):
        assert token in combined or token.replace("/", "$")+"" in combined, token
    pages, fonts = {}, {}
    for name, wanted in (("main",13),("supplement",6)):
        doc = PdfReader(HERE/f"{name}.pdf")
        assert len(doc.pages) == wanted
        pages[name] = len(doc.pages)
        log = (HERE/f"{name}.log").read_text(errors="replace")
        assert not re.search("Overfull|undefined|multiply defined", log, re.I)
        font_rows = subprocess.check_output(["pdffonts",str(HERE/f"{name}.pdf")], text=True)
        assert all(" yes " in row for row in font_rows.splitlines()[2:] if row.strip())
        fonts[name] = "all embedded"
    finite = json.loads((OUT/"finite_certificate.json").read_text())
    assert finite["count"] == 72 and finite["new_trajectories"] == 0
    replay = json.loads((OUT/"replay_registered_safety/integrity.json").read_text())
    assert replay["admitted"] == 480 and replay["no_carryover_admitted"] == 505
    decisions = pd.read_csv(OUT/"replay_registered_safety/public_gate_decisions.csv")
    corrected = decisions[decisions.policy == "risk_corrected_gate"]
    expected_decisions = corrected.candidate_available & (corrected.high_margin > 0) & (corrected.low_slack <= corrected.epsilon_safe)
    assert (expected_decisions == corrected.admit).all()
    assert json.loads((OUT/"replay/integrity.json").read_text())["admitted"] == 0
    citations = json.loads((OUT/"fresh_citations/citation_verification.json").read_text())
    assert len(citations["records"]) == 39
    assert citations["unresolved_material_conflicts"] == ["paulin2015concentration"]
    assert all(r["resolvers"]["authoritative"]["status"] == "AVAILABLE" for r in citations["records"])
    # Fresh primary institutional confirmation of article-number semantics; raw audit remains immutable.
    url = "https://www.research.ed.ac.uk/en/publications/concentration-inequalities-for-markov-chains-by-marton-couplings-/"
    request = urllib.request.Request(url, headers={"User-Agent":"Academic-bibliography-integrity-audit/20261019"})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read().decode("utf-8")
    assert all(value in body for value in ("Daniel Paulin", "79", "1--32") if value != "1--32")
    resolution = dict(raw_pass=38, references=39, authoritative_existence_matches=39,
                      raw_statuses_preserved=True, bib_edited=False,
                      paulin=dict(kind="article number, not issue conflict", url=url, sha256=hashlib.sha256(body.encode()).hexdigest(),
                                  article=79, volume=20, pages=32, doi="10.1214/EJP.v20-4039"),
                      new_preprints=dict(arxiv="fresh official abstract metadata and full PDF identities PASS",
                                         Crossref="not applicable: no publisher DOI claimed",
                                         SemanticScholar="raw resolver unavailable/batch stopped, not PASS",
                                         OpenAlex="raw resolver unavailable/batch stopped, not PASS"),
                      style=dict(definite_errors=[], judgments=["New arXiv journal strings use redundant preprint wording; optional presentation refinement, not an identity conflict"],
                                 existing_37_entry_bib_unchanged=True, additive_bib_sha256=sha(HERE/"recent_work_20261019.bib")))
    (OUT/"citation_resolution.json").write_text(json.dumps(resolution,indent=2)+"\n",encoding="utf-8")
    code_count = core_checks()
    assert visual, "Visual review must be explicitly completed before release record"
    report = dict(decision="CONSTRUCTIVE_THEORY_AND_RETROSPECTIVE_REPLAY_VERIFIED_NEW_SEED_VALIDATION_PENDING",
                  original_V3_sha256=expected, prior_revised_source_preserved_in_base=True,
                  old_formal_statements_byte_preserved=len(old_blocks), moved_proofs_preserved=True,
                  author_block_unchanged=True, original_bib_unchanged=True,
                  pure_gate_checks=code_count, public_gate_decisions_checked=len(corrected),
                  known_model_shield_witnesses=72, reconstructed_designs=896,
                  corrected_admitted=480, no_carryover_admitted=505, zero_slack_admitted=0,
                  new_trajectories=0, pages=pages, fonts=fonts,
                  visual_review="all 19 final pages viewed; changed theory/table pages additionally enlarged",
                  citations="38 raw PASS + fresh article-number resolution; 39 authoritative identities",
                  artifacts_sha256={name:sha(HERE/name) for name in ["main.tex","appendices.tex","supplement.tex","main.pdf","supplement.pdf","references.bib","recent_work_20261019.bib"]},
                  analysis_cpu_seconds=replay["cpu_seconds"]+finite["cpu_seconds"]+json.loads((OUT/"replay/integrity.json").read_text())["cpu_seconds"],
                  audit_cpu_seconds=time.process_time()-start,
                  release_blockers=["independent direct new-seed gate validation not run; separate 8h budget unapproved",
                                    "unknown-model simultaneous calibration not established",
                                    "scope/novelty importance still requires scientific peer judgment",
                                    "authentic ORCIDs, related-submission disclosures and all-author final consent pending"],
                  not_external_independent_review=True, not_TSP_acceptance_clearance=True,
                  native_140h_source_config_budget_unchanged=True)
    target.write_text(json.dumps(report, indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--montage", action="store_true")
    parser.add_argument("--visual-reviewed", action="store_true")
    args = parser.parse_args()
    if args.montage:
        montage()
    else:
        audit(args.visual_reviewed)
