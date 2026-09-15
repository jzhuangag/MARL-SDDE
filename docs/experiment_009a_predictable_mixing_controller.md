# EXP-009A preregistration: predictable mixing-certificate controller

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-30
- Verification Status: PREREGISTERED
- Version Label: exp009a_preregistration_v1

## Question

Can a past-only mixing certificate safely select the decorrelation gap and,
after charging its pilot cost, remain competitive with oracle and robust
baselines?

EXP-009A isolates mixing estimation in the scalar EXP-008C model. Curvature,
the operator bound \(L\), delay, and conditional sharing are treated as known.
They will require separate estimation audits.

## Frozen statistical design

- Scenarios:
  \(p\in\{0.5,0.9,0.98\}\),
  \(\rho\in\{0,0.9\}\), homogeneous delay \(D\in\{0,2\}\).
- Seeds: 128 fixed seeds per scenario and policy.
- Candidate participation:
  \(q\in\{1,2,4,8,16,32\}\).
- Total resource budget: 20,000 units.
- Server overhead: 8 units per update.
- An update with gap \(b\) and participation \(q\) costs \(b+8+q\).
- Online pilot: 2,048 raw regime transitions, charged to the budget.
- One-sided pilot failure probability: \(\alpha=0.01\).
- Initial delayed errors: one.
- Additive innovation scale: \(\sigma=0.2\), with aggregate variance
  \[
  \Omega_q=\sigma^2\{\rho+(1-\rho)/q\}.
  \]

For the symmetric two-state chain, stay indicators are i.i.d.
Bernoulli\((p)\). The online policy forms the exact one-sided
Clopper--Pearson upper bound \(p^+\) from pilot transitions. All decisions are
made after the pilot and use no exploitation samples.

The gap is the smallest \(b\ge1\) satisfying

\[
\tfrac12(2p^+-1)^b\le\mu/(4L).
\]

For each candidate \(q\), the policy computes the EXP-008E sharp rate step and
the registered risk surrogate

\[
\widehat R(q)
=c_{\rm sharp}(q)^{\lfloor U(q)/(2D+1)\rfloor}
+\frac{\eta(q)^2\Omega_q}{1-c_{\rm sharp}(q)},
\qquad
U(q)=
\left\lfloor
\frac{B-B_{\rm pilot}}{b+8+q}
\right\rfloor.
\]

It selects the minimizing \(q\). Ties go to smaller \(q\).

## Frozen policies

1. **online-UCB:** estimated \(p^+\), charged pilot.
2. **oracle:** true \(p\), no pilot charge.
3. **i.i.d.-naive:** assumes \(p=0.5\), no pilot.
4. **worst-mixing:** assumes \(p=0.98\), no pilot.
5. **oracle-\(q=1\):** true \(p\), fixed \(q=1\).
6. **oracle-\(q=32\):** true \(p\), fixed \(q=32\).

The two fixed-\(q\) baselines retain oracle mixing information and are
therefore strong participation baselines.

## Recorded endpoints

- pilot upper bound and coverage;
- selected \(b,q,\eta\), charged updates, and theorem surrogate;
- exact homogeneous covariance spectral radius for the selected policy;
- final squared error, maximum squared error, and divergence;
- paired final-error ratios using matched scenario/seed identifiers.

## Preregistered gates

1. **Certificate coverage.** At least 98.5% of online pilots satisfy
   \(p\le p^+\).
2. **Conditional exact safety.** On every covered pilot, the online selected
   step has exact homogeneous covariance radius below one.
3. **Online trajectory safety.** The online policy has zero divergence in all
   1,536 scenario/seed runs.
4. **Naive failure is detectable.** At \(p=0.98\), the i.i.d.-naive policy is
   exactly unstable in at least 90% of runs.
5. **Oracle competitiveness.** In every scenario, the median paired
   online-to-oracle final-error ratio is at most 5.
6. **Robust-baseline improvement.** For \(p\le0.9\), online-UCB has lower
   median final error than worst-mixing in at least six of eight scenarios.
7. **Participation responds to correlation.** Averaged over \(p,D\), the
   median selected \(q\) under \(\rho=0.9\) is no larger than under
   \(\rho=0\).

## Decision

Failures of coverage or conditional exact safety invalidate the estimator.
Failures confined to efficiency retain it as a safe proof-of-concept but
prevent a near-oracle controller claim. This two-state audit cannot by itself
establish a general finite-state mixing estimator theorem.
