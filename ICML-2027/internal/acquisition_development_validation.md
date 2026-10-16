# Acquisition-value development validation

## Material Passport

- Origin: academic-research-suite / experiment-agent.
- Date: 2026-10-13.
- Status: VERIFIED by an independent byte-identical replay.
- Frozen source and preregistration: 1d92557.

## Decision

All ten frozen development gates and exact replay pass. Authorize a separate
fresh 64-seed confirmation lock, preserving the controller and all thresholds.

The new algorithm chooses whether to acquire calibration before observing it,
then chooses q and spacing over the full catalogue. Its causal scalar update
gain minimizes quadratic Lyapunov risk and is shared by the efficient fixed-q
comparators. Static q may depend on every public budget/mixing configuration.

## Results

All 360 contexts, 16 fresh seeds, and 46,080 method rows are retained.

- Ratio to public-budget-wise efficient fixed q: 0.947742, paired-seed 95%
  interval [0.947592, 0.947892], a 5.2258% geometric terminal-MSE reduction.
- Ratio to the same-cost admissible oracle: 1.010125 [1.009965, 1.010285].
- Ratio to always purchasing calibration with the same efficient learner:
  0.985886 [0.985845, 0.985926].
- The uniform futility theorem skips acquisition in 80 contexts. The skipped
  controller never calls its sensor and exactly matches the full-budget fallback.
- Interior-qualified contexts: 84; correct complete-catalogue q selection: 99.7024%.
- Probed joint parameter coverage: 99.1964%; all-action risk coverage: 99.7098%.
- On the joint event every candidate true risk lies in its declared interval.
- Every method is finite and satisfies both message and total-actor budgets.

The advantage is over the full declared population; it is not universal
cellwise dominance. Public known mixing, a stationary Gaussian reward process,
and a constant-feature critic define the theorem and experimental scope.

## Reproduction

| File | SHA-256 |
|---|---|
| endpoints.csv | f0e1696ac8ece0c4f26143d674eb27a22807e626941319d9a2dee3a7c10a8fb8 |
| certificates.csv | 2161b1da0cc72c4a7307e02008c33f49a58ec4152620a7e8c893664a01e3a1ff |
| summary.json | 02f425d6804d4d88208932786eae07a8ca8c653c27f6c8279793d8162ef4012f |

All 11 statistical fallacy categories were checked. No cells or seeds were
removed. Seed aggregation, not row count, defines paired inference. The
parameter-coverage theorem and the paired-seed performance intervals are
distinct claims. Context-specific outcomes and calibration costs remain in
the raw tables. No nonlinear task-level generalization is inferred.

## Why the strengthened design matters

CPU-002 passes its original global-single-q gates. Its stronger budget-wise
diagnostic ratio 1.066058 is retained in theory/acquisition_gate.md. CPU-003
repairs the cost mechanism through a proved pre-observation acquisition test
and lower-cost exact sufficient-statistic calibration. CPU-002 thresholds and
results are unchanged, and none of its data is promoted to fresh confirmation.
