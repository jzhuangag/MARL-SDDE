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
- Every charged-resource axis uses five ticks, `0.00`, `0.25`, `0.50`, `0.75`,
  and `1.00`, with the same two-decimal formatter.
- The two single-column figures are generated at the final column width rather
  than being generated at double-column width and then reduced.

## Reproducibility

Two independent executions of the versioned plotting scripts produced
byte-identical PDFs:

| Figure | SHA-256 |
|---|---|
| `exp010b_joint_actions.pdf` | `8A571BCEE0459D2BC7A50CB313F5EB0D688C9124C6A648DDB1D38EA1FDF3A114` |
| `exp016b_risk_reduction.pdf` | `92A09B2296FBACFD3D05E9C1B08D50A494EC43EEB32829069DFDED76B0B963AE` |
| `convergence_curves.pdf` | `D204CC6872E350BA0102391D05ECAAD0E6C9E7A2594FA2463876CE791DD7DEEA` |
| `marl_multienvironment_returns.pdf` | `0EF4CD217D601DE9FB7E53ACBE6F4F8313F0BCA88970B819A2572877028C54B0` |

## Manuscript verification

- Canonical IEEEtran/`latexmk` build: pass.
- Final PDF: 13 letter-size pages.
- Final PDF SHA-256:
  `C45B47F2E7E2D47B11CE0872869EE0586614D38DFBE7BAD6F877E12AE9899B3C`.
- Undefined citations: 0.
- Undefined references: 0.
- Overfull boxes: 0.
- LaTeX warnings: 0.
- Rendered pages 9 and 10 were visually inspected at 180 dpi.  All legends,
  five-tick resource axes, panel titles, confidence bands, and captions are
  readable and unclipped.
- Full relevant regression suite: 80 passed.
- `git diff --check`: pass.

The Codex built-in standalone compiler could not initialize its platform
directories on this Windows host; the project-owned `build.ps1` completed the
canonical multi-pass IEEEtran build successfully.
