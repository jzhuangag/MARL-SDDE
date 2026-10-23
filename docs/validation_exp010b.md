# EXP-010B validation: affine Markov-TD finite-time certificate

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-30
- Verification Status: VERIFIED
- Version Label: validation_v1

## Result

- **ID**: EXP-010B-affine-finite-time-certificate
- **Type**: preregistered finite-gap affine Markov-TD theorem calibration
- **Status**: completed and exactly reproduced
- **Primary rows**: 1,152 policy runs from 32 fresh paired seeds
- **Numerical result**: 3/3 gates pass
- **Scientific result**: 6/6 gates pass
- **Overall preregistered result**: **PASS**

Theorem 4 removes the conditional-centering and Jacobian--innovation
orthogonality requirements from the affine TD bound.  It retains the
innovation inside the stale-iterate telescoping identity and produces a
finite-time mean-square bound with a stochastic residual and a finite-mixing
bias floor.

## Preregistered gates

| Gate | Observed result | Verdict |
|---|---:|---|
| Certificate validity | all selected \(a_\delta>0\), \(0<c_{\rm aff}<1\) | PASS |
| Execution validity | 1,152/1,152 valid; 0 divergences | PASS |
| Algebraic reproduction | maximum error \(2.22\times10^{-16}\) | PASS |
| Finite-time nonvacuity | bound beats no update in 12/12; minimum 106 updates | PASS |
| Correlation response | \(q\) decreases in 6/6 cells, strictly in 6/6 | PASS |
| Mixing response | median gap ratio \(980/53=18.49\) | PASS |
| Delay response | delayed step nonincreasing in 6/6 cells | PASS |
| Empirical upper calibration | 99% upper mean below bound in 12/12 | PASS |
| Bound informativeness | bound/mean below \(10^3\) in 12/12 | PASS |
| Overall | 3/3 numerical and 6/6 scientific | **PASS** |

## Selected joint actions

| \(\kappa\) | \(D\) | \(\rho\) | \(q\) | \(b\) | \(\eta\) | updates |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 16 | 64 | 0.0103 | 1,523 |
| 0 | 0 | 0.9 | 1 | 50 | 0.00194 | 2,327 |
| 0 | 8 | 0 | 4 | 56 | 0.00544 | 2,000 |
| 0 | 8 | 0.9 | 1 | 50 | 0.00194 | 2,327 |
| 0.9 | 0 | 0 | 32 | 250 | 0.0785 | 447 |
| 0.9 | 0 | 0.9 | 1 | 183 | 0.0149 | 680 |
| 0.9 | 8 | 0 | 4 | 205 | 0.0390 | 600 |
| 0.9 | 8 | 0.9 | 1 | 183 | 0.0149 | 680 |
| 0.98 | 0 | 0 | 16 | 1,185 | 0.133 | 106 |
| 0.98 | 0 | 0.9 | 1 | 924 | 0.0503 | 137 |
| 0.98 | 8 | 0 | 4 | 1,036 | 0.0998 | 122 |
| 0.98 | 8 | 0.9 | 1 | 924 | 0.0503 | 137 |

The theorem-backed rule selects an average \(q=12.67\) under independent
sources and \(q=1\) in every high-sharing cell.  Delay reduces the independent
action to \(q=4\), while increasing temporal persistence raises the median
paid gap by a factor of 18.49.

## Bound calibration

The proved joint-action bound is conservative but finite:

- bound/empirical-mean ratio: median 20.77, maximum 228.98;
- 99% bootstrap upper mean/proved-bound ratio: maximum 0.305;
- all twelve bounds are below the corresponding no-update error;
- all twelve 99% upper empirical means are below the theorem bound.

The largest relative looseness occurs for fast-mixing independent data, where
the observed error is already very small.  Under slow mixing and high sharing,
the bound/mean ratio falls to approximately 3.50.  Thus coverage is not
obtained merely by an infinite or numerically vacuous bound.

## Controller audit

The affine theorem closes the correctness gap, but optimizing a conservative
upper bound need not optimize observed error.  In the
\((\kappa,\rho,D)=(0,0,0)\) cell, the theorem selects \(q=16\); its mean error
is 1.47 times the \(q=32\) theorem-certified endpoint, with a paired-bootstrap
95% interval \([0.905,2.262]\).  The difference is not statistically resolved
at 95%, but it prevents a claim of empirical near-oracle participation.

Conversely, intermediate participation is materially valuable in delayed
independent cells: the selected \(q=4\) rule uses only 0.4%--1.2% of the
\(q=32\) endpoint mean error.  Under high sharing, the theorem selects
\(q=1\), avoiding redundant messages and stale updates.

The correct claim is therefore:

- the controller minimizes a proved finite-time certificate;
- it responds correctly and reproducibly to correlation, mixing, and delay;
- it is not claimed to minimize the unknown empirical TD error.

## Proof-scope decision

The completed result applies to the predictably decorrelated algorithm: the
joint chain advances by a charged gap before each retained update.  It is a
finite-time affine Markov-TD theorem and includes cross-agent dependence
through both \(K_q\) and \(\Omega_q\).

It does not prove the same rate for the aggressive unthinned recursion.  A
Poisson-equation argument remains an optional stronger extension, not a
blocker for the finite-gap algorithm.  Likewise, general online estimation of
the mixing certificate and the SDDE-to-discrete approximation remain separate
tasks.

## Reproducibility

An isolated full rerun used the same 32 seeds and 2,000 bootstrap
replications.  All eight primary artifacts matched byte-for-byte under
SHA-256.  The machine-readable hashes are stored in
`results/affine_finite_time_certificate/reproduction_hashes.json`.

## Validated decision

Promote Theorem 4 and EXP-010B into the main line.  The next CPU priorities
are:

1. derive a correlation-limited speedup lower bound;
2. decide whether the paper assumes a supplied mixing certificate or proves a
   general predictable estimator;
3. calibrate theorem-selected actions on one additional external linear-TD
   family before beginning the nonlinear GPU benchmark.

