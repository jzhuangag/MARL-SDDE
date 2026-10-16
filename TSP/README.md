# IEEE Transactions on Signal Processing manuscript workspace

This directory isolates the positive journal paper built from the project's verified correlation, mixing, delay, and finite-budget results.

The publication thesis is:

> Under correlated multi-agent Markov data, a learning system should jointly control participation, temporal spacing, and step size, and should pay to identify the dependence regime only when the remaining learning horizon can amortize that information cost.

The main paper is `main.tex`, its principal proofs are in `appendices.tex`, and routine resource-geometry derivations are packaged through `supplementary.tex`.
The verified bibliography is `references.bib`; the compiled artifacts are `main.pdf` and `supplementary.pdf`.
Publication-facing prose is contribution-led and does not reproduce the project's exploratory chronology.
Internal provenance, completion tasks, and evidence boundaries are kept under `internal/`.

## Build

Run `powershell -ExecutionPolicy Bypass -File build.ps1` from this directory.
The script regenerates the controlled-study figures from verified repository artifacts, retains the frozen confirmatory MAPPO figure, compiles both PDFs, and writes rendered page images under `tmp/rendered/` and `tmp/supplementary-rendered/` for visual inspection.

Run `..\.venv\Scripts\python.exe tools\verify_citations.py` for a fresh authoritative citation check and `..\.venv\Scripts\python.exe -m pytest tests\test_manuscript_evidence.py -q` for manuscript-to-evidence regression tests.

## Evidence used in the first integrated draft

- EXP-007A: correlation-limited effective participation in a seven-state linear temporal-difference task.
- EXP-010B: a finite-time affine delayed Markov temporal-difference certificate and joint selection of participation, spacing, and step size.
- EXP-016B: an independent 192-seed formal confirmation of the finite learning-value threshold and the learning-aware fallback rule.
- TSP-CURVE-001: a disjoint 16-seed development and 64-seed confirmation study of parameter-error and discounted-return-estimation convergence curves.
- TSP-MARL-CONF-001: a preregistered 48-run MAPPO confirmation with eight new training seeds and eight disjoint probe seeds on PettingZoo MPE `simple_spread_v2`.

The frozen source artifacts remain in their original repository locations and are not copied or altered here.

## Current integrated artifact

- Thirteen full IEEE double-column pages, including the principal proof appendices and 36 verified references, plus a one-page routine-proof and estimand supplement.
- Five theorem statements, five propositions/corollaries, and aligned proofs or proof reductions, including an epochwise moving-policy critic-tracking bridge.
- Five figures generated from frozen or seed-separated repository evidence, including deterministic team-return learning curves.
- No unresolved citation, cross-reference, or BibTeX warning; no overfull box or visible layout defect in the rendered manuscript.
- Anonymous author and funding placeholders remain for the authors to fill before submission.
