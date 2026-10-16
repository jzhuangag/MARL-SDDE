# Arrival-fresh MPE development audit

## Decision

The commit-barrier interface is executable and exactly charged, but the current
HARL SimpleSpread estimator does not yet isolate a freshness-specific benefit.
No confirmation, formal, or GPU run is authorized.  The next CPU design must
reduce policy-gradient variance and compare every fresh barrier against an
equal-cost extra-birth batching control.

## Executable timing and accounting

The runner uses distinct owner policy blocks and a single in-flight packet per
agent.  Each birth estimator averages two independent, clipped Monte Carlo
policy-gradient trajectories.  At a selected completion event, the centralized
trainer freezes commits, collects two independent trajectories under the
current joint policy, fuses their mean with the birth mean, applies the owner
update, and releases the barrier.  All actor transitions and serialized barrier
service time are charged; the hard transition budget is never exceeded.

The upstream HARL checkout remains pinned at
`b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`.  The owned runner SHA-256 is
`6b9268b7af6d355f5b6f7836dc6b19f42880b29c93505d9f1f20e2308e01ce22`.

## Four-seed return comparison

All modes receive 61,200 actor transitions per run.  The strong periodic
baseline chooses one phase per service profile from four complete phase runs;
it does not change phase per seed.

| Service profile | LSFF mean final return | vs never | vs always | vs best fixed periodic |
|---|---:|---:|---:|---:|
| balanced | -103.1730 | +3.8464 | +0.2368 | +0.3392 |
| heterogeneous | -100.9439 | +1.5460 | -2.5573 | -1.2859 |

The balanced mean is mildly positive, but the heterogeneous result fails the
central broad-robustness requirement.  Individual seeds also reverse direction.
A transition-only resource queue did not fix this: its balanced and
heterogeneous means were -104.9844 and -101.7274, both below the corresponding
dual-resource LSFF means.  Thus the heterogeneous failure is not explained
solely by pricing a wall-clock resource that was only reported rather than
hard-capped.

## Freshness-specific value audit

The more important failure is mechanistic.  In the three-agent experiment,
the observable strategic-bias component is roughly 250 times smaller than the
inflated Monte Carlo variance component:

| Profile | Mean bias-square upper | Mean birth-variance upper | Bias share of `A` | Bias share among selected refreshes |
|---|---:|---:|---:|---:|
| balanced | 1.727e-5 | 4.313e-3 | 0.399% | 0.234% |
| heterogeneous | 1.549e-5 | 4.411e-3 | 0.350% | 0.174% |

The selected refreshes are even more variance-dominated than the average
packet.  Their observed birth/fresh discrepancy coverage is 100%, but that
does not establish strategic-bias coverage or a return benefit.  A six-agent
heterogeneous smoke raises the bias share only to about 1.54%, while mean
return still decreases.

Consequently, the current barrier mainly buys an additional trajectory batch.
Attributing its benefit to asynchronous strategic freshness would be a
confounded claim.  The current comparison also lacks an extra-birth control
that purchases the same number of trajectories at packet birth without a
barrier.

## Required redesign before confirmation

The unified paper question remains viable, but the standard-task interface
must satisfy all of the following before new seeds are frozen:

1. use a learned action-independent critic, a larger trajectory batch, or an
   exact/tabular gradient to reduce ordinary policy-gradient variance;
2. log the predictable variance/bias decomposition and require a material
   strategic-bias share in the target population;
3. add always and fixed-period extra-birth batching baselines with the same
   actor-transition costs as their fresh-barrier counterparts;
4. compare LSFF against never, always-fresh, always-extra-birth, best fixed
   fresh period, and best fixed birth-augmentation period;
5. apply wall-clock queue prices only in a registered dual-budget experiment
   that actually imposes or reports the corresponding Pareto constraint;
6. use fresh development seeds after the estimator architecture is fixed, then
   freeze separate confirmation seeds and gates.

This is a useful stop decision rather than a paper result.  It prevents a
batching gain from being mislabeled as evidence for the proposed ICML story.
