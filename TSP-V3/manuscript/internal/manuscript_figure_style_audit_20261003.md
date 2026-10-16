# TSP-V3 experimental-figure style audit — 2026-10-03

## Scope

This audit covers every experimental figure included in the TSP-V3 manuscript:

- `figures/exp010b_joint_actions.pdf`;
- `figures/exp016b_risk_reduction.pdf`;
- `figures/convergence_curves.pdf`;
- `figures/marl_multienvironment_returns.pdf`.

The source data, estimands, confidence intervals, and scientific outcomes were
not changed.  Only rendering, typography, panel geometry, and the displayed
resource-axis ticks were revised.

## Unified style

- All text and mathematical labels are embedded as Times New Roman or its
  italic variant; no sans-serif or STIX font remains in the four figure PDFs.
- The shared style fixes label, title, tick, and legend sizes; line weights;
  confidence-band alpha; spine weights; and gray-grid color, alpha, and width.
- The Lyapunov-controlled method and every fixed participation value use a
  consistent color and line style across figures.
- Every multipanel figure uses equal subplot boxes.  Figs. 4 and 5 share the
  exact 7.15-by-2.25-inch canvas and identical subplot margins.
- Every `(a)`--`(d)` panel label is centered below its panel's x-axis label;
  no panel identifier remains above a plotting area.
- Every charged-resource axis uses five ticks, `0.00`, `0.25`, `0.50`, `0.75`,
  and `1.00`, with the same two-decimal formatter.
- The two single-column figures are generated at the final column width rather
  than being generated at double-column width and then reduced.

## Reproducibility

Two independent executions of the versioned plotting scripts produced
byte-identical PDFs:

| Figure | SHA-256 |
|---|---|
| `exp010b_joint_actions.pdf` | `DB508835B79F4E7EA9AB68D54D2C75C56339A91AC731D30CDEADB4A3D04C9016` |
| `exp016b_risk_reduction.pdf` | `92A09B2296FBACFD3D05E9C1B08D50A494EC43EEB32829069DFDED76B0B963AE` |
| `convergence_curves.pdf` | `A0D4BD7C2026F460CD0B0DF8A91A1D15B839E57B33B54B3BEBF242E2F27E2441` |
| `marl_multienvironment_returns.pdf` | `081ED9680856159FB2033BA408740A1C7DAD4A85D7695464FEDB19E986BE0715` |

## Manuscript verification

- Canonical IEEEtran/`latexmk` build: pass.
- Final PDF: 13 letter-size pages.
- Final PDF SHA-256:
  `6FE44327EB90D286DE354C0698F49611FCE0082BEA09CEC4DD7D7D5D7BA46BFD`.
- Undefined citations: 0.
- Undefined references: 0.
- Overfull boxes: 0.
- LaTeX warnings: 0.
- Rendered pages 9--11 were visually inspected at 180 dpi.  All legends,
  five-tick resource axes, below-panel labels, confidence bands, and captions
  are readable and unclipped.
- Full relevant regression suite: 155 passed.
- `git diff --check`: pass.

The Codex built-in standalone compiler could not initialize its platform
directories on this Windows host; the project-owned `build.ps1` completed the
canonical multi-pass IEEEtran build successfully.
