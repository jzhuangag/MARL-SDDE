# TSP-MARL-DEV-002 analysis compatibility erratum

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-10
- Verification Status: CODE-COMPATIBILITY-ERRATUM
- Version Label: erratum_v1

The preregistered analyzer at commit `208333b` used `numpy.trapezoid`, which is
available in the local NumPy 2.x test environment but absent from the pinned
HPC4 NumPy 1.23 environment.  The first remote analysis attempt therefore
raised `AttributeError` before writing `gate.json`, `curve_summary.csv`, or
`return_curves.pdf`.

The only scientific-code change uses `numpy.trapezoid(values, grid)` when
available and otherwise falls back to `numpy.trapz(values, grid)`.  These names
implement the same composite trapezoidal rule.  No run artifact, seed,
inclusion rule, budget, gate,
threshold, horizontal grid, pairing rule, or metric definition changed.  The
failed empty/partial `analysis` directory is retained; the corrected first
scientific output is written to a distinct `analysis_v2` directory.

- Frozen incompatible analyzer SHA-256:
  `36b60621c680b0a6eab2798d8c434a93bb3c062664baf91a1d9b8648fcec4ad4`.
- Raw experiment lattice: 24 completed cells; no cell is rerun.
- Required post-change validation: targeted tests, full TSP tests, remote
  NumPy-1.23 synthetic analyzer test, then one corrected analysis and one
  byte-level replay.

This erratum is an execution-compatibility amendment, not an outcome-dependent
analysis amendment.
