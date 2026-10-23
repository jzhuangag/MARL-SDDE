# EXP-008D validation: proof-derived decorrelated safe step

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validation
- Origin Date: 2026-07-30
- Verification Status: VERIFIED
- Version Label: exp008d_validation_v1

## Outcome

The coarse Theorem 3 step is exactly safe but too conservative under delay.

- Numerical checks: 2/2 passed.
- Scientific gates: 4/5 passed.
- Exact cells: 72/72.
- Selected decorrelation gaps: \(1,9,47\) for
  \(p=0.5,0.9,0.98\).

## Safety

Every theorem step lay inside the exact decorrelated Markov-jump stability
region. The largest exact spectral radius was 0.979615, and every computed
theorem contraction coefficient was strictly below one. This is a direct
finite-state validation of the proved total-variation and RMS-delay bound.

## Nonvacuity failure

The preregistered nonvacuity gate required at least 66 cells with a
theorem-to-exact ratio of at least 5%. Only 57 passed. The overall ratio range
was 4.28%--85.86%.

The failure is entirely delay driven:

- homogeneous delay zero: 44.1%--85.9%, 36/36 above 5%;
- homogeneous delay two: 4.28%--9.13%, 21/36 above 5%.

Thus the decorrelation argument is not the source of vacuity. The loss comes
from expanding the delayed cross term and then applying a pathwise
\(\|e-\eta a\|\le2\|e\|\) bound.

## Corrective theory action

Applying the \(L_2\) triangle inequality before expanding the delayed cross
term yields the stronger proved condition

\[
\sqrt{1-2\eta\mu_\delta+\eta^2K_\delta}
+\eta^2L^2\tau_{\rm rms}<1.
\]

It recovers the exact fresh-sample sufficient condition when delay is zero and
removes the factor-four delayed cross-term bound. This new condition was
derived after, and is not evaluated as part of, EXP-008D. It requires a new
preregistered audit.

## Decision

Retain the coarse condition as a valid fallback lemma, not as the main
algorithm. The sharp \(L_2\) condition and its rate-optimal interior step are
the new controller target.

## Artifacts

- `decorrelated_boundaries.csv`
- `summary.json`
- `fig1_theorem_tightness.png`
- `fig2_best_participation.png`
