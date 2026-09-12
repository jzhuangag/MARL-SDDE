# EXP-008B validation: exact Markov-jump boundary

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validation
- Origin Date: 2026-07-30
- Verification Status: VERIFIED
- Version Label: exp008b_validation_v1

## Outcome

EXP-008B is numerically valid but does not activate the intended temporal
mechanism strongly enough to support a mixing-adaptive claim.

- Numerical checks: 4/4 passed.
- Scientific gates: 3/6 passed.
- Registered exact cells: 36/36 completed.
- Maximum covariance-operator dimension: 72.
- Maximum eigen-residual: \(7.18\times10^{-15}\).

The result is retained as a negative control. The frozen model and thresholds
were not changed after inspection.

## Numerical checks

1. Direct enumeration of common draws, idiosyncratic draws, and sharing masks
   agreed with the moment-form conditional operator to maximum absolute error
   \(1.89\times10^{-15}\).
2. Every Markov, i.i.d., and mode-conditioned first-moment boundary was finite,
   positive, and bracketed by radii on the correct sides of one.
3. At \(p=0.5\), every Markov boundary equaled the corresponding stationary
   i.i.d. boundary to the stored numerical precision.
4. Every \(q=1\) boundary and scalar step was exactly invariant to the
   conditional pair-sharing parameter.

These checks validate the implementation of Theorem 2's mode-conditioned
covariance operator for this two-regime model.

## Scientific results

### Temporal persistence was weak

At \(p=0.98\), the Markov-to-i.i.d. boundary ratio ranged from 0.96898 to
1.00223. No cell met the preregistered ratio-at-most-0.8 criterion.
Persistence therefore changed the boundary by at most about 3.1% in the
adverse direction and was slightly beneficial in some delayed cells.

This does not refute the Markov-jump theorem. It shows that persistence alone
is not a sufficient experimental mechanism: the selected conditional TD
operators do not produce sustained local expansion strong enough to make
mixing the active stability margin.

### The uninflated scalar rule was safe but vacuous

The same-time scalar rule was stable in all 36 exact Markov cells. Its selected
step was only 2.98%--8.53% of the exact Markov boundary, and its largest exact
spectral radius was 0.99648. Thus the rule's safety here comes with severe
conservatism; EXP-008B does not justify presenting it as an efficient adaptive
algorithm.

### Agent-count interpretation requires care

The registered direction comparing \(q=3\) with \(q=1\) under low and high
conditional sharing passed, but the saturation threshold passed in only three
of six slices. More importantly, the \(D=2\) comparison changes the delay
composition from \((2)\) to \((0,1,2)\). That slice cannot isolate an
agent-count effect. Only the all-zero-delay slices are clean participation
comparisons. Subsequent exact sweeps must use the same homogeneous delay for
every agent when testing \(q\), with heterogeneous delay reported separately.

### Mean and mean-square margins were close

The exact mean-square boundary was 84.4%--96.6% of the exact
mode-conditioned first-moment boundary. No cell met the preregistered
mean-square-at-most-80%-of-mean threshold. In this model, multiplicative
second moments matter, but they are not the dominant source of instability.

## Decision

1. Retain EXP-008B as a verified negative control and implementation audit.
2. Do not use it as evidence for temporal-correlation adaptation.
3. Do not promote the current scalar formula as the main algorithm; it is far
   too conservative in this audit.
4. Run a new, separately preregistered Markov-modulated TD stress test with a
   locally expanding conditional regime and a positive stationary mean.
5. In the new test, keep delay homogeneous within each participation slice so
   \(q\) is not confounded with delay composition.

## Artifacts

- `markov_jump_boundaries.csv`
- `direct_validation.json`
- `summary.json`
- `fig1_temporal_boundary_ratio.png`
- `fig2_agent_correlation_boundary.png`
- `fig3_scalar_safety.png`
