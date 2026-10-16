# TSP-V3 manuscript integrity and style audit — 2026-10-01

## Deliverable

- Source: `TSP-V3/manuscript/main.tex`
- Compiled PDF: `TSP-V3/manuscript/main.pdf`
- PDF SHA-256: `DC7F682B9C589A3B6E5C7CAD0EBCE0861EFE1E3F37715F5FF4A3B9249B7DB80C`
- Length: 13 IEEE two-column pages
- Abstract: 214 words
- Keywords: 5

## TSP narrative audit

- The introduction follows a single contribution-led progression: policy-evaluation problem and applications; correlation, mixing, budget, and delay challenges; four progressively related method classes and their precise gap; the proposed learning-aware Lyapunov controller; four contributions; and the paper roadmap.
- The simulations section is organized around five theory-facing questions rather than an experiment log.
- Publication prose distinguishes regime-wise noninferiority from unknown-regime mixture superiority. It does not claim universal dominance over every fixed participation action.
- Internal preregistration, gate, and failure-process language is excluded unless it is needed to define an estimand, comparator, or evidence split.

## Evidence trace

The MaMuJoCo claims were checked against `TSP-V3/results/marl_mamujoco_calibrated_confirmation_20260930/gate.json` and `summary.csv`:

- 80 fresh-seed runs covering the controller and fixed `q` in `{1,2,4,8}` under independent and shared coupling;
- `q=8` selected in 8/8 independent runs and `q=2` in 8/8 shared runs;
- relative AUC against the regime-matched fixed envelope: -0.541906% and -0.996873%, both within the prespecified 2% noninferiority margin;
- equal-mixture AUC gain against the strongest single fixed action: 8.340817%, with a one-sided 95% lower bound of 1.107360%, 7/8 positive pairs, and exact sign-test `p=0.03515625`;
- equal-mixture final-return gain: 4.636745%;
- maximum probe message fraction: 0.768%;
- maximum selection-overhead fraction: `3.083640240627682e-7`.

The corresponding MPE study uses the same complete fixed catalogue and reports regime-wise deviations of -0.357% and -0.889%, plus a 2.319% equal-mixture AUC gain over the strongest single fixed action.

## Build and visual inspection

- Canonical `latexmk`/IEEEtran build: pass.
- Undefined citations or references: 0.
- Overfull boxes: 0.
- LaTeX and BibTeX warnings: 0.
- The PDF remains 13 pages after the prose revision.
- All 13 rendered pages were visually inspected; the introduction, simulations, four-panel return figure, appendix transition, and reference list have no clipping, overlap, blank spill page, malformed glyph, or unreadable label.

## Citation and bibliography gates

- Fresh authoritative citation verification: 37/37 pass, with no unresolved material conflict (`citation_verification_20261001.json`).
- IEEE bibliography-style audit: pass (`bibliography_style_audit_20261001.md`).
- Cited keys: 37; BibTeX entries: 37; missing, uncited, or duplicate entries: 0.
- Leading whitespace in TeX source: 0 lines; forbidden `\qquad`: 0; long-form `Equation`/`Figure` cross-references: 0.

## Tests

- Full relevant `TSP/tests` and `TSP-V3/tests` suite: 80 passed in 41.78 seconds.
- `git diff --check`: pass.
