# EXP-016A analysis plan v2

This document supersedes `exp016a_analysis_plan.md` for any future EXP-016A
work. The original plan remains immutable provenance.

## Static eligibility before outcomes

All core cells keep their original scenario, regime, budget, policy, and seed
assignments. Gray-zone cells remain outside core theorem gates. Negative
controls remain explanatory only and cannot rescue positive gates.

For every mechanism comparison, the runtime must first record whether the two
policies have identical probe, commit, action, delay, and budget paths. A cell
with identical paths is a consistency cell, not an effect cell.

## Revised G6

G6a is a deterministic theorem/runtime audit. Every above-`B_S` cell must use
the frozen bidirectional thresholds, registered likelihood, stopping boundary,
`q/b`, true public delay, and true dual-budget accounting. Both analytic
directional bounds must be <= `0.025`. Threshold mismatches and hidden-state
leakage must be exactly zero.

G6b is empirical calibration. Low-to-high and high-to-low are aggregated by
direction across all preregistered above-`B_S` seed blocks using exact
one-sided binomial/Clopper-Pearson simultaneous bounds. The aggregate
directional upper bound must be <= `0.025`. Per-cell errors and intervals are
reported, but a 64-seed per-cell rare-event CI is not a mandatory pass gate.
Any single-cell observed error rate above `0.10` triggers failure or manual
implementation audit.

## Revised practical-effect gates

G4 keeps the mandatory positive high-regime direction over all above-`B_S`
high cells. The 2% practical-effect claim applies only to the outcome-free
subset whose analytic expected relative gain is at least `0.02`; the current
subset size is `108`.

G5 is first a deterministic theorem compliance audit against each scenario's
`epsilon_safe`. Empirical safety intervals are calibration evidence and may
not be formulated as an impossible per-cell CI gate.

G8 applies only to the outcome-free learning-value-active subset, defined by
different intended learning-aware and information-only plans plus analytic
expected relative gain at least `0.03`. The current active subset size is
`0`.
An empty subset is an immediate novelty failure.

G9 applies only to delay-active `D=12` cells where no-delay planning and true
delay planning differ. G10 applies only to message-active or environment-active
cells where the corresponding single-budget ablation changes the intended
plan. Identical-path cells are reported as consistency checks.

## Resampling and reporting

CRN pairing does not increase effective sample size. The bootstrap resampling
unit remains the complete seed block, and cross-budget shared random prefixes
must not be counted as independent repetitions. CVaR90 with 64 seeds is
secondary only. Analytic risks may be used for thresholds, oracle joins, and
feasibility audits, but never as observed MSE.
