# EXP-011B: dual-anytime correlation/mixing controller

## Material Passport

- Artifact: preregistration
- Role: end-to-end CPU test of predictable unknown-\((p,\rho)\) control
- Formal seeds: 32
- Formal base seed: 20261231
- Pilot exclusion: four implementation seeds with base 20261211 were inspected
  only to debug semantics and set realistic gates; they are excluded from all
  formal evidence
- Pre-evidence correction: the first formal process was stopped before it
  wrote result artifacts when proof audit showed that fixed-sample
  Clopper--Pearson intervals were insufficient under adaptive sample sizes

## Question

Can a low-complexity controller safely and usefully adapt participation,
decorrelation gap, and scalar step size when both temporal mixing and
cross-agent sharing are unknown?

The registered model exposes two Bernoulli indicators:

1. whether the two-state Markov regime stays at a transition, with probability
   \(p\); and
2. whether an observed agent pair shares its update source, with probability
   \(\rho\).

This collision/share indicator is observable in the registered model.  The
experiment does not claim that every multi-agent environment directly exposes
it; learning a general latent correlation certificate remains separate.

## Frozen controller

For each Bernoulli stream, use the beta-binomial mixture likelihood ratio

\[
M_n(u)=
\frac{B(S_n+\tfrac12,n-S_n+\tfrac12)}
{B(\tfrac12,\tfrac12)u^{S_n}(1-u)^{n-S_n}}.
\]

Under the true parameter, \(M_n\) is a nonnegative martingale.  Ville's
inequality makes
\(\{u:M_n(u)<1/\alpha_s\}\) a confidence sequence under arbitrary adaptive
sample sizes.  Assign \(\alpha_s=.005\) to each stream; a union bound gives
joint time-uniform coverage of at least .99.  Upper endpoints are rounded
upward to grids of .002 for \(p\) and .02 for \(\rho\).  The next block's
\((q,b,\eta)\) is selected using only past data and both rounded upper bounds.

The initial paired pilot has 128 trials.  Each trial costs
\(h+2=10\), so the charged pilot cost is 1,280 of the 20,000-unit budget.
Thereafter, an action with \(q\ge2\) naturally supplies one share observation
per update.  Unspent block resource is converted to charged two-agent probes.
No covariance matrix, inverse, preconditioner, or actor--critic is used.

## Frozen scenarios and policies

\[
p\in\{.5,.9,.98\},\qquad
\rho\in\{0,.5,.9\},\qquad
D\in\{0,2\}.
\]

Each of 18 cells receives 32 fresh seeds.  Registered policies are:

1. dual-anytime UCB controller;
2. correlation-blind controller using the identical mixing-confidence
   mechanism, pilot charge, and resource budget but fixing \(\rho=0\);
3. a true-\((p,\rho)\), no-pilot informed controller reference used only for
   reporting.  It is not a dynamic-programming or empirical oracle.

Exact mode-conditioned covariance propagation evaluates every time-varying
action.  The informed-reference comparison is descriptive because EXP-009D
already rejected a uniform near-oracle guarantee near \(p=1\).

## Pre-registered decision rule

Both validity gates must pass, and at least four of five scientific gates must
pass.

### Validity gates

1. Joint simultaneous coverage across both streams is at least 97.5% of
   formal dual-controller runs.
2. On every simultaneously covered run, every action that executes at least
   one update has exact covariance spectral radius strictly below one.

### Scientific gates

1. Median selected participation at \(\rho=.9\) is strictly smaller than at
   \(\rho=0\).
2. At \(\rho=.9\), dual adaptation has smaller median exact error than the
   correlation-blind policy in at least five of six \((p,D)\) cells.
3. At \(\rho=0\), the largest cellwise dual/blind median-error ratio is at
   most 1.5.
4. At \(p=.98\), the median final persistence upper bound is smaller than its
   initial value in at least five of six \((\rho,D)\) cells.
5. The median final correlation upper bound is smaller than its initial value
   in at least nine of 18 cells.

## Interpretation boundary

A pass establishes a predictable, simultaneously certified
correlation/mixing-adaptive controller in the registered observable-sharing
model.  It does not establish a general latent-correlation estimator,
near-oracle efficiency uniformly as \(p\to1\), or nonlinear MARL performance.
