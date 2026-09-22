# T-021 fixed-q nonlinear validation design

## Status

This is an outcome-free CPU design audit, not a preregistration. It creates no
trajectory, allocates no seed, and authorizes no GPU job. Any execution must
use a new experiment identifier; EXP-017B remains permanently stopped.

## Design principle

The nonlinear bridge should test the theorem's observable mechanism directly,
not attempt to learn an online participation policy. Participation is fixed in
`q in {1,4,16,32}`. The common/private stream construction preserves each
agent's fixed-policy marginal law. Mixing is public or independently
certified, and every message/environment cost is charged.

### Layer A: frozen-parameter gradient mechanism

For each task and predeclared network checkpoint, freeze the parameter vector.
Generate Markov blocks under a fixed behavior policy and record per-agent TD
gradients without applying an update. For each seed-block compute:

- the trace variance of the averaged gradient;
- pairwise gradient covariance;
- the normalized factor relative to `q=1`;
- the predicted factor `rho+(1-rho)/q`;
- mixing- and delay-stratified residuals.

The seed, not an individual gradient block, is the inferential cluster. Blocks
increase measurement precision within a seed but must not be counted as
independent formal replications.

### Layer B: fixed-q learning under dual budgets

Run the same frozen q catalogue under message-binding and
environment-binding rays, with zero, jitter, and bursty delays. Report terminal
prediction error, normalized learning AUC, CVaR90, actual message bytes,
environment steps, wall time, and agent transitions. This layer tests whether
the mechanism predicts resource-matched learning, not whether a controller can
select q online.

## Prospective primary families

1. **Correlation saturation:** the normalized averaged-gradient variance
   follows `rho+(1-rho)/q`, and the marginal benefit of increasing q is smaller
   at high rho.
2. **Dual-budget phase:** the selected fixed q under an environment-binding
   ray is no smaller than under the paired message-binding ray.
3. **Delay degradation:** fixed-q progress or error gain is nonincreasing as
   the registered effective delay increases.

Familywise alpha is 0.05 with three one-sided families. A practical ratio of
1.05 is frozen for power calculation. An implementation pilot may estimate
only the paired log-ratio standard deviation. For SD 0.10/0.15/0.20, 90%
power requires 49/110/196 seed-level replications. The maximum prospective N
is 192; if the pilot upper uncertainty bound implies more than 192, stop rather
than alter alpha or the practical effect.

## Mandatory preregistration gates

- exact marginal-law invariance and common-stream correlation tests;
- no gradient update in Layer A and no cross-cell checkpoint leakage;
- finite outputs and exact dual-budget accounting;
- cluster-level analysis with no block pseudoreplication;
- a direct variance-identity endpoint, not only best-q counts;
- fixed familywise correction, effect threshold, seed registry, and
  runner/analyzer hashes;
- no pilot outcome used to alter tasks, rho, q, delays, or rays;
- a failed primary gate stops formal claims without replacement.

The arithmetic remains `O(qd)` per averaged gradient and uses no Hessian,
covariance inverse, or preconditioner.

