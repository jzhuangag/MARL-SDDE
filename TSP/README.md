# IEEE Transactions on Signal Processing manuscript workspace

This directory isolates the positive journal paper built from the project's verified correlation, mixing, delay, and finite-budget results.

The publication thesis is:

> Under correlated multi-agent Markov data, a learning system should jointly control participation, temporal spacing, and step size, and should pay to identify the dependence regime only when the remaining learning horizon can amortize that information cost.

The main paper is `main.tex`, the proof supplement is `appendices.tex`, the verified bibliography is `references.bib`, and the compiled manuscript is `main.pdf`.
Publication-facing prose is contribution-led and does not reproduce the project's exploratory chronology.
Internal provenance, completion tasks, and evidence boundaries are kept under `internal/`.

## Build

Run `powershell -ExecutionPolicy Bypass -File build.ps1` from this directory.
The script regenerates the two manuscript figures from verified repository artifacts, compiles the paper, and writes rendered page images under `tmp/rendered/` for visual inspection.

Run `..\.venv\Scripts\python.exe tools\verify_citations.py` for a fresh authoritative citation check and `..\.venv\Scripts\python.exe -m pytest tests\test_manuscript_evidence.py -q` for manuscript-to-evidence regression tests.

## Evidence used in the first integrated draft

- EXP-007A: correlation-limited effective participation in a seven-state linear temporal-difference task.
- EXP-010B: a finite-time affine delayed Markov temporal-difference certificate and joint selection of participation, spacing, and step size.
- EXP-016B: an independent 192-seed formal confirmation of the finite learning-value threshold and the learning-aware fallback rule.

The frozen source artifacts remain in their original repository locations and are not copied or altered here.

## Current integrated artifact

- Seven IEEE double-column pages, including the proof appendices and 32 verified references.
- Four theorem statements with aligned proofs or proof reductions.
- Two figures regenerated from frozen repository evidence.
- No unresolved citation, cross-reference, BibTeX, or layout warning.
- Anonymous author and funding placeholders remain for the authors to fill before submission.
