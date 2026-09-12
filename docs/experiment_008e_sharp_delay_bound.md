# EXP-008E preregistration: sharp \(L_2\) delayed bound

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-30
- Verification Status: PREREGISTERED
- Version Label: exp008e_preregistration_v1

## Question

Does the sharp \(L_2\) delayed condition repair EXP-008D's delay
conservatism while preserving exact Markov-jump safety?

## Frozen cells and decorrelation

All 72 EXP-008D cells, decorrelation target, and selected smallest gaps are
unchanged. No model, persistence, correlation, delay, or participation value
is added or removed.

## Frozen sharp condition

Define

\[
c_0(\eta)=1-2\eta\mu_\delta+\eta^2K_\delta,\qquad
f(\eta)=\sqrt{c_0(\eta)}+\eta^2L^2\tau_{\rm rms}.
\]

The sharp stability boundary is the first positive solution of \(f(\eta)=1\)
after the small-step stable interval. The reported safe-boundary step is
\((1-10^{-8})\) times that root.

The theorem-rate step is the unique numerical minimizer of \(f(\eta)^2\) on
the connected safe interval. Both are deterministic scalar solves; no exact
Markov boundary is used by either solve.

## Numerical checks

1. Every returned root must bracket \(f(\eta)=1\) to absolute error at most
   \(10^{-9}\).
2. Every rate step must lie strictly inside the sharp safe interval and have
   no larger theorem coefficient than either endpoint.
3. All exact covariance boundaries and spectral radii must meet the EXP-008D
   residual and bracketing checks.

## Scientific gates

1. **Sharp-boundary exact safety.** All 72 safe-boundary steps have exact
   Markov covariance radius below one.
2. **Sharp-boundary nonvacuity.** At least 66 of 72 steps are at least 8% of
   the exact boundary, and no step exceeds the exact boundary.
3. **Rate-step exact safety.** All 72 theorem-rate steps have exact covariance
   radius below one.
4. **Theorem envelope validity.** For every cell,
   \[
   \rho(\mathfrak L_{\eta_{\rm rate}})^{2D+1}
   \le c_{\rm sharp}(\eta_{\rm rate})+10^{-8}.
   \]
5. **Delay improvement.** In every delayed cell, the sharp safe-boundary step
   is strictly larger than the coarse EXP-008D step.

## Decision

A 5/5 pass promotes the sharp scalar solve to the provable controller. A
safety or envelope failure invalidates the proof. A failure confined to
nonvacuity keeps the result as a theorem envelope but prevents an algorithmic
efficiency claim.
