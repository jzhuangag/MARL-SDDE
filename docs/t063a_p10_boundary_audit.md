# T-063A P10 boundary audit (post-result, non-rescuing)

## Purpose

This is a diagnostic audit of the single failed T-063A formal gate.  It does
not change the preregistered threshold, the formal classification, or any
scientific endpoint.  T-063A remains a formal failure because P10 was frozen
as mandatory.

## Observed event

At `rho=0`, the formal run contains 1,536 unique seed-by-task probe blocks
(512 seeds times three tasks), each with 96 independent fingerprint blocks.
The match-count distribution is:

| matches in a 96-block seed/task probe | count |
|---:|---:|
| 0 | 1,472 |
| 1 | 63 |
| 2 | 1 |

The unique maximum is the Seaquest block with master seed `202608056481`.
Its rate is `2/96 = 0.0208333333333333`, exceeding the frozen P10 limit
`0.0200000000000000`.  Therefore the stored primary summary correctly has
`P10_independent_collision=false` and `formal_authorized=false`.

## Calibration diagnostic

The frozen exact-kernel upper bound for an independent four-transition
fingerprint is `c_L = 0.0007716049382716049`.  Under the corresponding
Binomial(96, c_L) diagnostic null,

* `Pr(K=0) = 0.9285763550`;
* `Pr(K=1) = 0.0688365483`;
* `Pr(K>=2) = 0.0025870966`;
* across 1,536 seed/task blocks, the expected number with `K>=2` is
  `3.9737804`;
* the probability of seeing at least one `K>=2` block is `0.9812943`.

Thus the observed single `K=2` block is not evidence that the fingerprint
law or the controller failed.  It does show that a maximum-per-block rate
gate of `0.02` is not calibrated for a 1,536-block formal workload.  A future
experiment may replace this with a prospective family-wise exact-binomial or
aggregate collision gate, but such a replacement cannot be applied to
T-063A retrospectively.

## Scope boundary

All efficacy gates (aggregate, taskwise, delay, breadth, true-rho proximity,
direction, and calibration) remain as stored.  This audit is explanatory
only; it is not a post-hoc pass, confidence interval, or new formal evidence.
