# EXP-008A preregistration: exact lifted mean-square boundary

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-30
- Verification Status: PREREGISTERED
- Version Label: exp008a_preregistration_v1

## Question

Does the low-complexity scalar joint step from EXP-007D lie inside the exact
mean-square stability region of heterogeneous-delay temporal-difference
learning when the same-time cross-agent model is retained and transition pairs
are resampled independently over time?

This deterministic experiment audits the theorem, not Monte Carlo
trajectories.  The exact lifted state is

\[
X_{k+1}=M_k(\eta)X_k,
\]

and the covariance operator is

\[
\mathcal L_\eta(P)=\mathbb E[M_k(\eta)PM_k(\eta)^\top].
\]

The exact first positive mean-square boundary is the upper endpoint of the
connected stable interval satisfying \(\rho(\mathcal L_\eta)<1\).

## Registered cells

- MRP, features, discount, agent ordering, and heterogeneous delay profiles:
  unchanged from EXP-007A--D.
- Agent counts: \(q\in\{16,32\}\).
- Registered maximum delays: \(D\in\{0,8,32\}\).
- Cross-agent correlations: \(\rho\in\{0,0.9\}\).
- Total exact cells: 12.

The operator is applied matrix-free using delay-group counts and the finite-MRP
map

\[
\mathcal R_{\rm d}(Y)=\mathbb E[HYH^\top],
\qquad
\mathcal R_{\rm o}(Y)
=
\rho\mathcal R_{\rm d}(Y)+(1-\rho)AYA^\top.
\]

No dense Kronecker matrix is permitted for the registered \(D=32\) cells.
Small synthetic cells use both dense and matrix-free implementations as an
independent numerical check.

## Registered comparisons

For each cell save:

1. exact lifted critical step \(\eta_{\rm exact}\);
2. spectral radius at the EXP-007D scalar step
   \(\eta_{\rm joint}\);
3. \(\eta_{\rm joint}/\eta_{\rm exact}\);
4. exact delayed mean boundary \(\eta_{\rm mean}\);
5. \(\eta_{\rm exact}/\eta_{\rm mean}\).

The scalar formula remains unchanged:

\[
\eta_{\rm joint}
=
\left[
\eta_{\rm mean}^{-1}+K(q,\rho)/(2\mu)
\right]^{-1}.
\]

## Preregistered gates

All seven gates must pass.

1. **Independent numerical implementation.**  On registered small synthetic
   operators, the dense Kronecker radius and matrix-free radius differ by at
   most \(10^{-8}\).
2. **Boundary validity.**  All 12 cells return a finite positive boundary;
   the radius immediately below it is less than one, the radius immediately
   above it is greater than one, and every eigensolver residual is at most
   \(10^{-7}\).
3. **Scalar-rule safety.**  In all 12 cells,
   \(\rho(\mathcal L_{\eta_{\rm joint}})<1\).
4. **Nonvacuous scalar tightness.**  The ratio
   \(\eta_{\rm joint}/\eta_{\rm exact}\) is at least 0.25 in at least 10 of
   12 cells and never exceeds one.
5. **Correlation shrinks the exact region.**  For every \((q,D)\), the
   \(\rho=0.9\) exact boundary is at most half its \(\rho=0\) value.
6. **Agent-count saturation without delay confounding.**  At \(D=0\),
   increasing \(q\) from 16 to 32 enlarges the exact boundary by at least 15%
   under independence but by at most 5% under \(\rho=0.9\).
7. **Mean stability is insufficient.**  In at least 10 of 12 cells,
   \(\eta_{\rm exact}\le0.5\eta_{\rm mean}\).

## Decision

- A 7/7 pass promotes the scalar rule as a conservative approximation to an
  exact heterogeneous-delay theorem under temporally independent sampling.
- Any scalar-safety failure withdraws the rule as a theorem candidate.
- A failure confined to tightness retains the exact mechanism but removes the
  low-complexity rule from the primary algorithm.
- Passing EXP-008A does not establish the Markov-data theorem; it authorizes
  the subsequent Markov jump and mixing analysis.

