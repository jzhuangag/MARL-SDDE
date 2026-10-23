# EXP-008D preregistration: proof-derived decorrelated safe step

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-30
- Verification Status: PREREGISTERED
- Version Label: exp008d_preregistration_v1

## Question

Does the proof-derived decorrelation-gap algorithm satisfy the exact
Markov-jump stability test, without fitting a safety constant to EXP-008C?

## Frozen design

Use all 72 EXP-008C cells unchanged. For the symmetric regime chain, let

\[
\lambda=2p-1,\qquad
\delta(b)=\tfrac12|\lambda|^b.
\]

The decorrelation target is fixed from Theorem 3 as

\[
\delta_{\rm target}=\frac{\mu}{4L},
\]

which preserves at least half of the stationary monotonicity margin:
\(\mu_\delta=\mu-2L\delta\ge\mu/2\). Select the smallest integer \(b\ge1\)
for which \(\delta(b)\le\delta_{\rm target}\). No gap or constant is adjusted
using EXP-008C's exact boundaries.

At used samples, the regime persistence is exactly

\[
p_b=\frac{1+\lambda^b}{2}.
\]

Set

\[
K_\delta=K_q+2L^2\delta(b),\qquad
\tau_{\rm rms}=
\left(q^{-1}\sum_i\tau_i^2\right)^{1/2}.
\]

The theorem step is the largest \(\eta\le1/L\) satisfying

\[
\eta(K_\delta+4L^2\tau_{\rm rms})
+\eta^3L^4\tau_{\rm rms}^2
<2\{\mu-2L\delta(b)\}.
\]

It is found by deterministic scalar bisection.

## Numerical checks

The EXP-008C operator checks remain mandatory. Additionally, the recorded gap
must be the smallest admissible integer, and substituting the returned step
into the theorem polynomial must not exceed its right-hand side.

## Scientific gates

1. **Exact safety.** The theorem step has exact covariance spectral radius
   below one in all 72 decorrelated cells.
2. **Strict theorem slack.** The computed contraction coefficient
   \(c(\eta)\) is below one in all 72 cells.
3. **Nonvacuity.** The theorem step is at least 5% of the exact decorrelated
   boundary in at least 66 of 72 cells and never exceeds it.
4. **Mixing adaptation.** The selected gap is strictly increasing over
   \(p=0.5,0.9,0.98\).
5. **Participation remains correlation limited.** Under \(\rho=0.9\), the
   exact decorrelated \(q=32\)-to-\(q=16\) boundary gain is at most 1.05 in
   every delay/persistence slice.

The 5% threshold is deliberately weak because Theorem 3 is a worst-case
bounded-operator result. Exact safety is the primary gate; efficiency will be
addressed by the predictable controller and stronger empirical baselines.

## Decision

- A full safety pass promotes Theorem 3's scalar root as the base provable
  algorithm.
- A safety failure invalidates either the implementation or theorem and must
  be resolved before any controller experiment.
- A nonvacuity failure retains the theorem only as a correctness envelope and
  prevents an efficiency claim.
