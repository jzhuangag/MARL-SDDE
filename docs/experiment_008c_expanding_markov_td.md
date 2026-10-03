# EXP-008C preregistration: locally expanding Markov TD stress test

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-30
- Verification Status: PREREGISTERED
- Version Label: exp008c_preregistration_v1

## Motivation fixed before execution

EXP-008B validated the exact Markov-jump implementation but its persistence
effect was at most about 3.1%. Every registered conditional regime mixed
contracting and nonnormal directions in a way that left the exact boundary
nearly unchanged. EXP-008C asks the sharper theorem question: when a
conditionally expanding TD regime is allowed but the stationary mean remains
strongly monotone, is an explicit mixing margin necessary and can a
spectral-gap inflation restore a useful low-complexity rule?

EXP-008B remains unchanged as the negative control.

## Frozen scalar TD construction

Use scalar features

\[
\phi(0)=1,\qquad \phi(1)=2,\qquad \gamma=0.9.
\]

The four TD Jacobians, ordered as \(00,01,10,11\), are

\[
(0.1,-0.8,2.2,0.4).
\]

The same two conditional emission distributions are retained:

\[
w_0=(0.05,0.85,0.05,0.05),\qquad
w_1=(0.05,0.05,0.85,0.05).
\]

Their conditional means are \(-0.545\) and \(1.855\), while the stationary
mean is \(A=0.655>0\). Thus one persistent regime is locally expanding, but
the stationary problem is strongly monotone. The emission sequence is a
finite-state Markov-modulated TD oracle; it is used as an exact stress test,
not claimed to be a complete environment benchmark.

The regime transition, conditional pair-sharing construction, and exact
mode-conditioned covariance theorem are unchanged from EXP-008B.

## Registered cells

- Agent counts: \(q\in\{1,2,4,8,16,32\}\).
- Homogeneous delays: \(\tau_i=0\) for every agent or \(\tau_i=2\) for every
  agent.
- Persistence: \(p\in\{0.5,0.9,0.98\}\).
- Conditional pair-sharing correlation: \(\rho\in\{0,0.9\}\).
- Total exact cells: 72.

Homogeneous delays remove the participation/delay-composition confound found
in EXP-008B.

## Frozen step rules

Let \(K(q,\rho)=\lambda_{\max}\mathbb E[\bar H_q^\top\bar H_q]\),
\(\mu=A\), and let \(\eta_{\rm mean}\) be the stationary homogeneous-delay
mean boundary.

The uninflated rule is

\[
\eta_{\rm iid}
=\left[\eta_{\rm mean}^{-1}+K(q,\rho)/(2\mu)\right]^{-1}.
\]

For the symmetric two-state regime chain, the nontrivial eigenvalue is
\(\lambda=2p-1\). Freeze the integrated-autocorrelation proxy

\[
\chi_{\rm gap}(p)
=\frac{1+\lambda}{1-\lambda}
=\frac{p}{1-p}.
\]

The proposed stress-test rule is

\[
\eta_{\rm gap}
=\left[
\eta_{\rm mean}^{-1}
+\chi_{\rm gap}(p)K(q,\rho)/(2\mu)
\right]^{-1}.
\]

No constants will be refit after observing exact boundaries.

## Numerical checks

The four EXP-008B implementation checks remain mandatory: direct enumeration,
valid bracketing and residuals, exact i.i.d. reduction at \(p=0.5\), and exact
\(q=1\) invariance to \(\rho\).

## Preregistered scientific gates

1. **Persistence activates instability.** At \(p=0.98\), the exact
   Markov-to-i.i.d. boundary ratio is at most 0.5 in at least 18 of 24
   \((q,D,\rho)\) cells.
2. **An i.i.d. rule is insufficient.** The uninflated rule is unstable in at
   least 12 of the 24 cells with \(p=0.98\).
3. **Gap-inflated safety.** The gap-inflated rule has exact Markov spectral
   radius below one in all 72 cells.
4. **Gap-inflated nonvacuity.** In at least 60 of 72 cells,
   \(\eta_{\rm gap}/\eta_{\rm exact}\ge0.20\), and the ratio never exceeds
   one.
5. **Conditional correlation does not improve stability.** For every
   \((q,D,p)\) with \(q\ge2\), the \(\rho=0.9\) exact boundary is no larger
   than the \(\rho=0\) boundary.
6. **Correlation-limited participation.** In every \((D,p)\) slice, the
   \(q=32\)-to-\(q=16\) boundary gain under \(\rho=0.9\) is at most 1.05 and
   is no larger than the corresponding gain under \(\rho=0\).
7. **Mean stability is insufficient.** In at least 48 of 72 cells, the exact
   mean-square boundary is at most 80% of the exact Markov first-moment
   boundary.

## Exploratory quantities, excluded from the pass decision

- the exact required inflation
  \[
  \chi_{\rm req}
  =\frac{2\mu}{K}
  \left(\eta_{\rm exact}^{-1}-\eta_{\rm mean}^{-1}\right);
  \]
- \(\chi_{\rm req}/\chi_{\rm gap}\);
- whether delay and persistence combine additively or multiplicatively in
  inverse-step space.

These quantities may motivate a later theorem or preregistered estimator, but
they cannot be used to redefine EXP-008C's gates.

## Decision

- A safe and nonvacuous gap rule promotes spectral-gap or integrated
  autocorrelation estimation as the low-complexity algorithmic target.
- If it is safe but vacuous, retain it only as a theorem envelope and develop
  a sharper predictable estimator in a new experiment.
- Any safety failure prevents the frozen formula from entering the main
  theorem.
- A failure of the persistence gate means the proposed stress mechanism is
  inadequate; it is retained and not silently replaced.
