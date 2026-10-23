# TSP-V3 manuscript integrity audit — 2026-09-30

## Deliverable

- Source: `TSP-V3/manuscript/main.tex`
- Compiled PDF: `TSP-V3/manuscript/main.pdf`
- PDF SHA-256: `0d8ce6894ed25023e51a4cd8207772e7fc376ef1fd44c774a3d14478b1be645c`
- Length: 13 IEEE two-column pages
- Abstract: 220 words
- Keywords: 5

## Evidence trace

The MaMuJoCo claims in the abstract, experiments, caption, and conclusion were checked against `TSP-V3/results/marl_mamujoco_calibrated_confirmation_20260930/gate.json` and `summary.csv`:

- 80 fresh-seed runs, comprising the controller and fixed `q` in `{1,2,4,8}` under independent and shared coupling;
- `q=8` selected in 8/8 independent runs and `q=2` in 8/8 shared runs;
- relative AUC against the regime-matched fixed action: -0.541906% (independent) and -0.996873% (shared);
- equal-mixture AUC gain against the strongest single fixed comparator: 8.340817%, one-sided 95% lower bound 1.107360%, 7/8 positive pairs, exact sign-test `p=0.03515625`;
- equal-mixture final-return gain: 4.636745%;
- maximum probe message fraction: 0.768%;
- maximum selection-overhead fraction: `3.083640240627682e-7`.

The multi-environment figure is generated only from versioned MPE and MaMuJoCo summaries. Its trace is in `figure_trace_20260930.json`; the figure SHA-256 is `760dc0bb16a94379bc61b06e128969351b88dd7db3cc1c5592ffff85ad862952`.

## Build and layout

- `latexmk`/IEEEtran build: pass.
- Undefined citations or references: 0.
- Overfull boxes: 0.
- LaTeX warnings: 0.
- BibTeX warnings/errors: 0.
- All 13 rendered pages were visually inspected. No clipped text, overlap, blank spill page, unreadable axis label, or malformed reference was found.
- The four return panels are arranged in one row so the multi-environment evidence remains legible without displacing theory or appendix content.

## Source and bibliography checks

- Cited keys: 37; BibTeX entries: 37; missing or uncited entries: 0.
- Duplicate keys, empty fields, and duplicate rendered references: 0.
- Fresh authoritative citation verification: 37/37 pass, with no unresolved material conflict (`citation_verification_20260930.json`).
- IEEE bibliography-style audit: pass (`bibliography_style_audit_20260930.md`).
- Leading whitespace in TeX source: 0 lines.
- Forbidden `\qquad` spacing: 0 occurrences.

## Tests

- Targeted confirmation/manuscript tests: 15 passed.
- Full relevant `TSP/tests` and `TSP-V3/tests` suite: 80 passed in 274.46 seconds.
- Figure generation and complete PDF compilation were rerun after the final prose edit.
