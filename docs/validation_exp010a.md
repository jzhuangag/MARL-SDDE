# EXP-010A validation: multistate certificate transfer

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-30
- Verification Status: VERIFIED
- Version Label: validation_v1

## Result

- **ID**: EXP-010A-multistate-certificate-transfer
- **Type**: preregistered seven-state, four-feature linear TD transfer
- **Status**: completed and exactly reproduced
- **Primary rows**: 1,152 policy runs from 32 fresh paired seeds
- **Numerical result**: 3/3 gates pass
- **Scientific result**: 4/5 gates pass
- **Overall preregistered result**: **PASS**

The experiment transfers the sharp homogeneous stability certificate from the
two-regime audit to a vector-valued TD model.  Every selected action had a
strict contraction certificate, every execution stayed within the charged
resource budget, and no trajectory diverged.

## Preregistered gates

| Gate | Observed result | Verdict |
|---|---:|---|
| Stochasticity/stationarity | maximum residual \(3.33\times10^{-16}\) | PASS |
| Certificate/accounting/numerics | 1,152/1,152 valid; 0 divergences | PASS |
| Independent TV reproduction | maximum difference \(2.72\times10^{-15}\) | PASS |
| Nonvacuity | minimum 98 updates; all steps positive | PASS |
| Correlation response | \(q\) decreases in 6/6 matched cells, strictly in 6/6 | PASS |
| Mixing response | median gap ratio \(973/53=18.36\) | PASS |
| Delay response | delayed step nonincreasing in 6/6 cells | PASS |
| Endpoint value | strict improvement over both endpoints in 7/12 cells | **FAIL** |
| Overall | 3/3 numerical and 4/5 scientific | **PASS** |

## Selected joint actions

| \(\kappa\) | \(D\) | \(\rho\) | \(q\) | \(b\) | \(\eta\) | updates |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 32 | 68 | 0.0525 | 1,230 |
| 0 | 0 | 0.9 | 1 | 50 | 0.00989 | 2,327 |
| 0 | 8 | 0 | 4 | 56 | 0.0267 | 2,000 |
| 0 | 8 | 0.9 | 1 | 50 | 0.00989 | 2,327 |
| 0.9 | 0 | 0 | 32 | 228 | 0.294 | 484 |
| 0.9 | 0 | 0.9 | 2 | 171 | 0.0699 | 723 |
| 0.9 | 8 | 0 | 4 | 166 | 0.0791 | 735 |
| 0.9 | 8 | 0.9 | 2 | 171 | 0.0699 | 723 |
| 0.98 | 0 | 0 | 32 | 1,266 | 0.943 | 98 |
| 0.98 | 0 | 0.9 | 2 | 973 | 0.311 | 130 |
| 0.98 | 8 | 0 | 4 | 924 | 0.298 | 137 |
| 0.98 | 8 | 0.9 | 2 | 973 | 0.311 | 130 |

The mean selected count is 18 under independent trajectories and 1.67 under
high pair sharing.  The delay profile changes the independent-data action from
\(q=32\) to \(q=4\), while high sharing already saturates participation at
\(q\le2\).

## Endpoint-gate diagnosis

The strict endpoint gate is the sole scientific failure, but it does not show
that joint selection is inferior.  The joint candidate set contains both
endpoints.  It selected \(q=1\) in two scenarios and \(q=32\) in three, making
strict improvement over the selected endpoint mathematically impossible in
those five cells.

Across all twelve scenarios:

- joint selection is never worse in mean error than the better endpoint;
- the maximum joint/better-endpoint mean-error ratio is exactly 1.0;
- it is strictly better than both endpoints in the remaining 7/12 scenarios;
- the strongest gains occur in delayed independent-data cells, where
  intermediate \(q=4\) has joint/\(q=32\) ratios between 0.024 and 0.034.

The preregistered gate remains marked **FAIL**.  A post-hoc weak-dominance
reinterpretation is reported only as diagnosis and is not substituted for the
frozen decision rule.

## Theorem-scope audit

The exact TV tensorization, monotonicity perturbation, aggregate curvature,
RMS-delay term, and sharp homogeneous contraction are theorem-backed.
EXP-010A verifies their nonvacuity and numerical implementation in dimension
four.

The finite-budget expression used to choose among certified actions is not a
generic Markov-TD risk theorem.  Its simple
\(\eta^2\Omega_q/(1-c)\) residual requires the conditional
centering/orthogonality assumption stated in the proof program.  Finite-gap TD
innovations need not satisfy that assumption.  Accordingly:

- claim **certified homogeneous stability**;
- claim **empirical finite-budget TD performance**;
- do not claim certified finite-sample affine TD error until a martingale or
  Poisson-equation argument handles the Markov innovation.

This distinction is central to manuscript correctness, not a cosmetic caveat.

## Reproducibility

An isolated full rerun used the same 32 seeds and 2,000 bootstrap
replications.  All eight primary artifacts matched byte-for-byte under
SHA-256.  The machine-readable comparison is stored in
`results/multistate_certificate_transfer/reproduction_hashes.json`.

## Fallacy and implementation audit

| Risk | Finding |
|---|---|
| Same-sample action selection | Actions use analytic certificates only; no seed outcome selects \(q,b,\eta\). |
| Survivorship bias | No seed or policy row was removed. |
| Uncharged thinning | Every gap, message, and server overhead term enters \(4+q+b\). |
| Correlation/delay confounding | Complete matched \((\kappa,\rho,D)\) tables are retained. |
| Strict-gate artifact | Reported explicitly; the frozen gate is not rewritten. |
| Exact-oracle leakage | Exact 49-state mixing is an offline certificate-transfer oracle, not represented as the online estimator. |
| Deep-MARL generalization | Not claimed. |

## Validated decision

The defensible linear mainline is now:

1. correlation-limited effective participation invalidates nominal linear
   speedup;
2. a scalar, matrix-free certificate jointly selects participation,
   decorrelation gap, and step size under RMS delay;
3. the homogeneous stability theorem transfers to multistate vector TD;
4. estimating near-unit mixing creates an unavoidable confidence/resource
   penalty;
5. the remaining mathematical bottleneck is affine Markov finite-time
   convergence, followed by an external nonlinear benchmark.

“State-adaptive” should not remain in the paper title: earlier state proxies
did not survive strong baselines, whereas correlation, mixing, participation,
and delay effects now have reproducible support.
