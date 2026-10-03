# Preregistration: coupled actor--critic oracle-headroom scan

Date: 2026-09-05.

Status: **frozen analytic protocol; no scan outcome has been generated.**

Frozen scenario-manifest SHA-256:
`528fbc597015981f8868e2d7aab567003d5b0574fadecff6b462be2c8b746ae0`.

## Purpose

This is the oracle-headroom kill gate in
`tracking_the_moving_game_mainline.md`.  It asks whether exact joint Lyapunov
control of the actor application scale `alpha` and critic correction scale
`beta` has enough value to justify deriving an observable Markov-game
algorithm.  It is not efficacy evidence and it uses latent moment state that a
practical learner cannot observe.

The protocol cannot rescue the failed Two Clocks MPE bridge.  It uses no old
pilot/formal endpoint, task outcome, seed or fitted coefficient.

## Frozen model and populations

The scan propagates exact first and second moments of a Gaussian
multi-block quadratic actor--critic system.  Every agent has a single in-flight
packet and a private periodic completion clock.  On completion, only that
owner's birth snapshot is refreshed.  Thus the stored packet is self-fresh in
its owner block and stale only in teammate blocks.

The primary population contains 128 scenarios:

- agents: `{2,4}`;
- symmetric off-diagonal game interaction: `{0.15,0.35}`;
- critic-to-actor bias sensitivity: `{0.3,0.6}`;
- initial critic tracking error: `{0.35,0.75}`;
- actor-induced critic-target sensitivity: `{0.4,0.9}`;
- service heterogeneity: `{mild,severe}`;
- actor/critic innovation variance: `{low,high}`.

There are 32 controls at initial critic error `0.75`: 16 set critic-target sensitivity to zero and 16 set
off-diagonal game interaction to zero.  The horizon is 64 packet-completion
events.  Initial moments, Hessians, clocks, noise, caps and all optimizer
settings are source constants in the frozen runner.

## Frozen methods

1. `coupled`: exact eventwise minimizer of the full two-variable Lyapunov QP;
2. `diagonal_online`: the same eventwise rule after setting only the QP cross
   curvature to zero;
3. `best_fixed`: a privileged per-scenario fixed `(alpha,beta)` pair selected
   by a 15-by-15 box grid followed by bounded continuous refinement from the
   six best grid points.

All methods receive the same event order, initial distribution and packet
noise law.  One actor and one critic opportunity are charged at every event
regardless of chosen scale.  The scan therefore cannot manufacture an
advantage by giving the dynamic rule more packets or critic calls.

## Frozen endpoints

For each method and scenario, report normalized expected composite Lyapunov
AUC, normalized terminal Lyapunov risk, mean `alpha`, mean `beta`, fraction of
jointly interior actions and mean event time.  Ratios below one favor the
coupled rule.  Aggregate ratios are geometric means over the 128 primary
scenarios; controls are never mixed into the main aggregate.

## Mandatory gates

| Gate | Frozen requirement |
|---|---|
| H1 | exactly 128 primary and 32 control scenarios; every output finite |
| H2 | every full QP is PSD and every selected action satisfies its box |
| H3 | coupled/best-fixed geometric AUC ratio `<=0.90` |
| H4 | coupled/diagonal-online geometric AUC ratio `<=0.95` |
| H5 | coupled strictly beats best-fixed in at least 70% and diagonal-online in at least 60% of primary scenarios |
| H6 | both mean actor and critic scales are positive in every primary scenario; at least 5% of primary actions are jointly interior in aggregate |
| H7 | all zero-target-motion controls have zero cross curvature and coupled/diagonal outputs agree to `1e-10` absolute tolerance |
| H8 | median coupled-over-diagonal AUC improvement is strictly larger at target sensitivity `0.9` than at `0.4` |
| H9 | continuous fixed-pair refinement never returns a worse value than its frozen grid start and every final pair respects the box |
| H10 | two clean executions produce byte-identical JSON |

Every gate is mandatory.  One failure stops this successor before sampled
CPU trajectories, formal seeds, GPU or HPC4.  Gates, scenarios and thresholds
must not be modified after outcome access.

## Authorized commands

Validation only, before commit:

```text
python -m experiments.clocked_async_mpg.run_coupled_actor_critic_headroom --validate
python -m pytest experiments/clocked_async_mpg/test_run_coupled_actor_critic_headroom.py -q
```

After this protocol and runner are committed, exactly two clean analytic runs
may be written under ignored result directories and compared byte for byte.
