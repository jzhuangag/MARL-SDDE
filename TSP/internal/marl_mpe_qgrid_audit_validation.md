# TSP-MARL-MPE-QGRID-AUDIT-001 validation report

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-16
- Verification Status: VERIFIED
- Version Label: mpe_qgrid_audit_final

## Scope and decision

This post-confirmation audit completes the fixed-participation comparator
catalogue for the existing PettingZoo MPE `simple_spread_v2` study.
It adds fixed `q=2` and fixed `q=4` on the eight registered seeds under both
independent and shared rollout coupling, while preserving all original
controller, fixed `q=1`, and fixed `q=8` results.

All four frozen audit gates passed.
The decision is **retain baseline-complete MPE transfer**.
The audit establishes comparator completeness; it does not relabel the added
post-confirmation cells as new confirmatory evidence.

## Main result

| Coupling regime | Fixed-q oracle | Controller gap to oracle |
|---|---:|---:|
| Independent | `q=8` | -0.3572% |
| Shared | `q=1` | -0.8892% |

Across the equal-weight regime mixture, fixed `q=8` was the strongest single
static action.
The Lyapunov controller improved return AUC over that strong static baseline
by 2.8117% while remaining within 1% of the complete fixed-q oracle in each
regime.

The mean return AUC values were:

| Regime | q=1 | q=2 | q=4 | q=8 | Lyapunov controller |
|---|---:|---:|---:|---:|---:|
| Independent | -96.9582 | -84.2150 | -76.5954 | -73.7762 | -74.0397 |
| Shared | -96.4704 | -99.8394 | -101.0496 | -102.5495 | -97.3282 |

These results support the intended claim: the controller tracks the value of
participation across dependence regimes rather than relying on either endpoint
of the participation catalogue.

## Gate ledger

| Gate | Result |
|---|---:|
| Finite, complete, exactly charged, clean pinned upstream | pass |
| Controller within 2% of fixed-q oracle in each regime | pass |
| All registered fixed q in `{1,2,4,8}` present | pass |
| Analysis replay byte-identical | pass |

The combined analysis contains 80 unique runs: 48 preserved predecessor runs
and 32 new fixed-q cells.

## Execution and integrity

| Slurm array | Frozen cells | Result |
|---|---:|---:|
| `1861812` | q=2, independent, seeds 96001--96008 | 8/8 `COMPLETED 0:0` |
| `1869605` | q=2, shared, seeds 96001--96008 | 8/8 `COMPLETED 0:0` |
| `1870882` | q=4, independent, seeds 96001--96008 | 8/8 `COMPLETED 0:0` |
| `1871619` | q=4, shared, seeds 96001--96008 | 8/8 `COMPLETED 0:0` |

All 32 result directories contain a bridge metadata file, progress record, and
checksum manifest.
All 32 checksum manifests passed and the fatal-log scan was empty.
Outputs remained under
`/scratch/jzhuangag/MARL-SDDE-TSP-MPE-QGRID-AUDIT-001`; no experiment output
was written to `/project` or `/home`.

## Reproduction

The corrected analyzer was run independently into `analysis_v2` and
`analysis_v2_replay`.
The curve table, PDF, and gate JSON were byte-identical.

- Gate JSON SHA-256: `a66662c829dcdc3f5a4c023d299ffb1eacd56197aad97459cf5bd83e0afeec73`.
- Curve CSV SHA-256: `8606ca6070728b4644e513491e4a9a0d27cc93838435cef8deb5cf0b800a2664`.
- Return-curve PDF SHA-256: `60ba047c4ce8e3290de3629c2016fd6361804ea6cb61a5028b47be1db6d7efd6`.

The original incomplete analysis directory and both infrastructure amendments
remain preserved as provenance.
