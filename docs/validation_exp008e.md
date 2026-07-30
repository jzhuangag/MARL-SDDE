# EXP-008E validation: sharp \(L_2\) delayed bound

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validation
- Origin Date: 2026-07-30
- Verification Status: VERIFIED
- Version Label: exp008e_validation_v1

## Outcome

EXP-008E passed all preregistered checks and gates.

- Numerical checks: 2/2 passed.
- Scientific gates: 5/5 passed.
- Exact cells: 72/72.
- Maximum eigen-residual: \(6.41\times10^{-15}\).

## Sharp boundary

Every sharp theorem boundary step was exactly mean-square stable. The largest
exact covariance radius was 0.9999999931, consistent with evaluating
\((1-10^{-8})\) times the theorem root.

The sharp-to-exact boundary ratio was 0.08097--0.99999999, and all 72 cells
exceeded the preregistered 8% threshold. In delayed cells the sharp step was
at least 1.8665 times the coarse EXP-008D step.

## Rate-optimal interior step

Every scalar minimizer of

\[
c_{\rm sharp}(\eta)
=
\left[
\sqrt{1-2\eta\mu_\delta+\eta^2K_\delta}
+\eta^2L^2\tau_{\rm rms}
\right]^2
\]

was exactly stable; the largest exact covariance radius was 0.980944.
Moreover, in every cell,

\[
\rho(\mathfrak L_{\eta_{\rm rate}})^{2D+1}
\le c_{\rm sharp}(\eta_{\rm rate})
\]

to numerical precision. The minimum stored envelope slack was zero rather
than negative.

## Decision

The sharp scalar solve is promoted to the base provable controller:

1. it is a one-dimensional computation;
2. it retains \(q\) through \(K_q\);
3. it retains temporal dependence through a decorrelation certificate;
4. it retains the complete delay profile through \(\tau_{\rm rms}\);
5. it uses no preconditioner, matrix inverse, or actor--critic.

The next gate is statistical rather than spectral: the mixing and curvature
inputs must be selected from past data with simultaneous confidence coverage
and charged probe cost.

## Artifacts

- `sharp_boundaries.csv`
- `summary.json`
- `fig1_sharp_boundary_tightness.png`
- `fig2_coarse_vs_sharp.png`
