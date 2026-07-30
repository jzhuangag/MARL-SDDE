# EXP-012B: unknown-baseline kernel certificate

## Material Passport

- Artifact: preregistration
- Role: continuous-state CPU validation of Theorem 8
- Formal seeds: 128 per scenario
- Formal base seed: 20270401
- Pilot exclusion: four seeds with base 20270311 were inspected before this
  file was frozen and are excluded from formal evidence

## Question

Can latent sharing be certified for continuous Markov observations when exact
sample equality has probability zero and the independent-source similarity
baseline is unknown?

## Frozen model

Three hidden chains evolve on the unit circle.  At each transition a chain
retains its value with probability

\[
\lambda\in\{0,.8,.96\}
\]

and otherwise refreshes from the uniform distribution.  The equivalent
two-state persistence passed to the existing controller is
\((1+\lambda)/2\in\{.5,.9,.98\}\).

Two agents independently select the common hidden chain with probability
\(\sqrt\rho\), where

\[
\rho\in\{0,.5,.9\}.
\]

The sharing masks and hidden source identities are unavailable to the
estimator.

The observed similarity is the periodic radial basis function kernel

\[
k(x,y)
=
\exp\left\{
-\frac{2\sin^2(\pi(x-y))}{.35^2}
\right\}.
\]

Its independent-source expectation is used only for evaluation, not by the
estimator.

## Frozen certificate

The mixing confidence sequence receives failure probability \(0.01/3\).
Before each block, its upper endpoint selects the smallest gap satisfying

\[
3(\lambda^+)^b\le .01.
\]

Each probe records a same-time agent similarity \(X_j\).  Starting with the
second probe, it also records a lagged control similarity between the previous
first-agent observation and the current second-agent observation.  Translation
invariance on the circle makes the stationary conditional control mean equal
to the unknown baseline \(c_k\).

The same-time upper confidence sequence and control lower confidence sequence
each receive failure probability \(0.01/3\).  Theorem 8 returns

\[
\rho^+
=
\left[
\frac{\vartheta^+-c_k^-}{1-c_k^-}
\right]_{[0,1]}.
\]

## Frozen resource and action protocol

- Total resource: 20,000.
- Initial mixing observations: 128.
- Decision block: 2,000.
- Pair-probe cost: \(b+8+2\).
- Unspent resource refines the mixing sequence.
- Final rounded confidence bounds select the unchanged scalar
  \((q,b,\eta)\) action at \(D\in\{0,2\}\).
- Exact covariance propagation at the true equivalent persistence and
  correlation evaluates safety.

The nine \((\lambda,\rho)\) cells each receive 128 fresh seeds.

## Preregistered decision

Both validity gates and at least four of five scientific gates must pass.

### Validity gates

1. Joint time-uniform coverage of \(\lambda,\vartheta_k,c_k,\rho\) is at
   least 97.5%.
2. Every updating final action on a jointly covered run has exact radius below
   one.

### Scientific gates

1. Median final \(\rho^+\) is non-decreasing in true \(\rho\) at each
   \(\lambda\).
2. At \(\lambda=0\), median \(\rho^+\) is at most .30 for \(\rho=0\), lies
   in \([.5,.9]\) for \(\rho=.5\), and covers .9 for \(\rho=.9\).
3. The fast-mixing median baseline lower bound is at least .02 in every
   correlation cell and no scenario-median lower bound exceeds the true
   baseline.
4. At \(\lambda=.96\), every cell has at least 50 median similarity probes.
5. At both delays, median selected \(q\) at \(\rho=.9\) is no larger than at
   \(\rho=0\) for all three mixing values, with a strict response in at least
   one mixing value across the two delays.

## Interpretation boundary

A pass establishes continuous-state hidden pair-sharing certification for a
bounded translation-invariant kernel and a control stream with predictable
mixing bias.  It does not cover arbitrary additive Gaussian common factors,
adversarial kernels, or a control stream whose stationary conditional mean
depends on the stored observation.
