# Multi-baseline q-grid analysis amendment 1

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-16
- Verification Status: VERIFIED-ANALYSIS-AMENDMENT
- Version Label: qgrid_analysis_amendment_1

## Failure and diagnosis

The first automated MPE q-grid analysis wrote its interpolated curve table and
then stopped before producing a figure or gate file.
The collector's shared `aggregate_curves` routine returns preaggregated columns
`team_return_mean`, `team_return_std`, and `seeds`.
The new plotting wrapper incorrectly attempted to aggregate a nonexistent raw
`team_return` column a second time, raising `AttributeError`.

No gate metric was evaluated, no result was admitted to the manuscript, and
the incomplete `analysis` directory is retained unchanged as provenance.

## Correction

The MPE plotting wrapper now consumes the documented preaggregated schema
directly.
A regression test constructs that exact schema and verifies that the PDF is
created.
The correction changes no record collection, return-AUC calculation,
comparator, threshold, seed, experimental outcome, or gate logic.

The clean reproduction writes to new `analysis_v2` and
`analysis_v2_replay` directories and must show byte-identical curve, figure,
and gate files before its decision is accepted.
