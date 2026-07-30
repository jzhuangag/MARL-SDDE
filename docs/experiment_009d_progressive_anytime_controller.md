# EXP-009D preregistration: progressive anytime-safe controller

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-30
- Verification Status: PREREGISTERED
- Version Label: exp009d_preregistration_v1

## Motivation

EXP-009A--C show that a one-shot 99% certificate is safe but statistically
inefficient when \(1-p\) is small. EXP-009D does not change the confidence
level or theorem. It reuses raw regime transitions observed between parameter
updates and refreshes the action only at block boundaries.

## Frozen design

- Same 12 scenarios, 128 seeds, feature/Jacobian model, additive noise,
  resource cost, candidate \(q\), and total budget 20,000 as EXP-009C.
- Initial observation-only pilot: 128 raw transitions.
- Decision/execution block budget: 2,000 resource units.
- At most ten decision blocks.
- Overall failure probability: \(\alpha=0.01\).

At decision \(m=1,2,\ldots\), allocate

\[
\alpha_m=\frac{\alpha}{m(m+1)}.
\]

Because \(\sum_{m\ge1}\alpha_m=\alpha\), a union bound makes the sequence of
one-sided Clopper--Pearson bounds \(p_m^+\) simultaneously valid with
probability at least 0.99 at all decision times.

The block-\(m\) action is the EXP-009C joint safe minimizer
\((q_m,b_m,\eta_m)\), computed from \(p_m^+\), the remaining total resource,
and data available before the block. The action is frozen within the block.
Stay indicators observed during the block update only \(p_{m+1}^+\).

For deterministic caching, the action calculation rounds \(p_m^+\) upward to
the nearest \(10^{-3}\). This can only make the mixing certificate more
conservative and does not affect the coverage event. The progressive oracle
uses true \(p\), receives the full 20,000-unit budget, and pays no pilot cost.

If the certificate is too weak to permit an update within a block, the block
is spent on observation-only regime transitions. This prevents an all-stay
initial pilot from causing an undefined or falsely safe action.

## Exact evaluation

Primary performance propagates the exact mode-conditioned affine covariance
state across time-varying block operators. Additive forcing and delayed-state
covariance are retained. Observation-only transitions mix the mode-conditioned
covariances without changing the parameter.

Registered policies:

1. progressive online anytime-UCB;
2. progressive oracle using true \(p\);
3. static EXP-009C online-UCB;
4. worst-mixing joint controller;
5. oracle fixed \(q=1\);
6. oracle fixed \(q=32\).

## Preregistered gates

1. **Anytime coverage:** at least 98.5% of seeds have simultaneous coverage at
   every executed decision.
2. **Conditional exact safety:** every progressive action on a fully covered
   seed has exact homogeneous covariance radius below one.
3. **Near-oracle expected risk:** every scenario-median progressive
   online/oracle exact expected-error ratio is at most five.
4. **Static-pilot improvement:** the largest scenario-median ratio is strictly
   below EXP-009C's 10.4640743243.
5. **Worst-baseline improvement:** progressive online has lower scenario
   median expected error than worst-mixing in at least ten of 12 scenarios.
6. **Progressive refinement:** at \(p=0.98\), the median final-block gap is
   strictly smaller than the median first updating-block gap.
7. **Participation response:** median selected \(q\) under \(\rho=0.9\) is no
   larger than under \(\rho=0\).

Safety failure invalidates the confidence/controller coupling. Efficiency
failure rejects near-oracle performance but preserves the static theorem.
