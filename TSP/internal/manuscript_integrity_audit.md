# TSP manuscript integrity audit

Audit date: 2026-09-09.

## Claims and numerical provenance

- EXP-007A statistics are checked against `docs/experiment_007a_linear_td_correlation.md`.
- EXP-010B statistics are checked against `docs/validation_exp010b.md`.
- EXP-016B formal statistics are read from `experiments/dependence_delay_linear/results/exp016b_formal_20260801/analysis/core_results.json`.
- TSP-CURVE-001 statistics are checked against `internal/convergence_curve_validation.json` and the committed aggregate curve data.
- The manuscript evidence regression suite checks all displayed primary percentages, confidence lower bounds, row counts, seed counts, scenario coverage, safety status, and gate booleans.
- Figures are regenerated from frozen repository evidence and fixed registered summaries; no manual image editing is used.
- The convergence confirmation uses 64 seeds disjoint from the 16 development seeds used to select the strong fixed-participation comparator.

## Theory integrity

- The affine delayed Markov result is the exact discrete-time guarantee used by the controller.
- The SDDE is labeled as a secondary local interpretation and is not used to replace the discrete proof.
- The adaptive information result preserves predictable dimension changes, irregular gaps, stopping, and both resource budgets.
- The learning-value theorem distinguishes actual risk from the computable certificate.
- The threshold theorem is restricted to the stated compact, separated, known-mixing class.

## Citation integrity

- All 32 cited records passed a fresh authoritative proceedings or Crossref DOI check.
- The report is `citation_verification_20260909.json`.
- There are no unresolved material metadata conflicts.

## Build and visual quality

- `latexmk` and BibTeX complete without undefined references, undefined citations, or overfull boxes.
- The final PDF has thirteen letter-size IEEE double-column pages.
- Every rendered page was visually inspected for clipping, broken formulas, unreadable figures, and reference overflow.
- Two consecutive clean figure-and-manuscript builds produced identical SHA-256 hashes for the PDF, all three PDF figures, and all three PNG figures.

## Required author input before submission

- Author list and affiliations.
- Funding and acknowledgments.
- Data/code availability language and any venue-required disclosure statements.
