# Independent TSP V3 repair — working copy,2026-10-18

## Material Passport

Human-authorized repair on2026-10-18; academic-research-suite, IEEE TSP writing,
publishable academic writing and IEEE Bib style routes. Original TSP-V3 source,
PDF, bibliography, figures and all scientific evidence remain untouched. This
directory is a separate working revision, NOT a submission release. No NEW
sealed-bridge experiment result is used as evidence here.

## Targeted patch ledger

Edits are local source patches, not full manuscript regeneration. Original
baseline SHA256 values:main.tex165adac70b67f2067964380351e8f94d129548b2b9da55007434a5821fba7e78;
appendices.texdfcec46942e3640bbe320b68b7aed42fc3944817a879e0fd6fc34f4d9a19e0cc;
references.bib9ce047e460c16da8ea3cbe1ed9bdf55b90b4c0b2eba19bbe71c571dd431a7690.
The Markdown-only ARS HTML-anchor mechanism is not inserted into LaTeX; Git
patches, baseline/revised hashes and the source diff preserve the exact scope.

| Audit item | First-pass source correction | Remaining verification |
|---|---|---|
| EXP016B denominator |35.24%/9.79% and subset32.94/43.72/30.13%; display converts saved gain/fallback to gain/information-only; old gates/results unchanged | Regenerate/inspect corrected figure and reconstruct exact values |
| T1 Gaussian KL | Explicit true-state/filter L2 recursions and finite expected information bound; no false pathwise Gaussian-history bound | Independent quantifier/optional-clock/Kalman proof check |
| T2 compactness/solver | Nonempty closed positive-margin certified subsets; empty-domain handling; separate numerical candidates from exact global solve | Verify retained saved actions admit margins; check all solver claims |
| T3 history/envelope | Deterministic delays, zero initialized increments, positive-part replacement; tracking requires uniform mixed-history drift | Check every robust/return/tracking implication and boundary |
| T4 safety | Compare to fallback CERTIFICATE, not unknown true baseline loss; no epsilon added to unbounded MSE | Check all theorem/abstract/interpretation wording and conditioning |
| T5 scope/cost |1/s risk expansion separate from constant-step TD; exact spectral calibration separate from rank-one aggregation | Check all remaining complexity and phase claims |

## Mandatory release gates still pending

Fresh citation-integrity check of all37 retained references and IEEE Bib style
audit; theorem/claim/source-evidence re-review; exact estimand definitions for
MPE mean normalized gain versus MaMuJoCo mixture-first gain; LaTeX reference and
line-style lint; complete compilation and visual inspection at final size;
author/coauthor/submission metadata from the human. No author information is
invented. Original audit remains valid as the original-version decision.

V3 repair must progress independently while the separate140h experiment runs.
Do not wait for it or import its incomplete outcomes. At unresolved proof gaps,
give an explicit repair/release decision, never a claim of readiness by fiat.

## Current second-pass handoff

See internal/second_pass_repair_review_20261018.md. The current PDF is13 pages,
abstract220 words/five keywords; all13 pages were rendered and inspected.
Build exits0 with zero undefined/overfull items. Equation-reference style is
normalized, risk/spacing figure labels are corrected, and36 saved actions have
strictly positive stable-domain containment margins. Original four hashes match.

Additional substantive repairs align the budget decomposition with its fallback,
make post-probe conditional certificates explicit, and close the missing phase
budget-perturbation assumption. These remain working proof/evidence repairs,
not submission clearance. A newly identified retained-update versus physical-
clock delay convention must be reconciled with each saved study, without
changing any old code/gate/result or launching native reruns. Full citation-
context and final protocol/sentence/metadata gates remain. Next work is that
model/accounting alignment, NOT another citation-only pass or experimental test.

## First-pass checks performed

- Full latexmk/PDF/BibTeX build exits0; final log has no undefined references or
  overfull boxes and three underfull warnings. The working PDF is14 pages,
  above the13-page target; shorten exposition without deleting proof-critical
  hypotheses. No format compression is authorized.
- Pages1/9/14 rendered and inspected: corrected figure is legible and matches
  the new denominator; final reference page is sparse. Remaining pages still
  require visual inspection after final page reduction.
- Main/appendix source has no leading whitespace or quadruple-space commands.
  Remaining legacy equation-reference style issues still require normalization.
- Fresh public-metadata verification of all37 retained references completed:
  36 raw PASS and one Paulin issue/article-number representation conflict,
  separately resolved using fresh institutional metadata; raw report preserved.
  No cached verification passes were reused. Index outages remain explicit;
  this is not a full-text citation-context or originality release.
- Original main.tex/appendices.tex/references.bib/main.pdf SHA values match all
  four original audit hashes. No original scientific result or gate was changed.
