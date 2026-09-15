## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: outcome-separated development design
- Origin Date: 2026-09-07
- Verification Status: FROZEN BEFORE HEADROOM OUTCOMES
- Version Label: pistonball_controller_headroom_v1

# Matched Pistonball controller-headroom matrix

## Purpose

This is the last development gate before an efficacy pilot. It asks whether
the complete signed-cache-queue Lyapunov decision has material learning value
over strong equal-resource schedulers after the learner and differentiable
score interfaces have separately qualified. It is not pilot or formal
evidence, and its two seeds `79006` and `79007` are permanently excluded from
those populations.

## Frozen system

Every run uses 20 distinct SiLU actors, a centralized SiLU critic,
decentralized evaluation, 4,096 launches, rollout horizon four, 327,680 exact
actor transitions, maximum receipt delay eight, actor step `0.01`, critic step
`0.0003`, randomized Pistonball physics, deterministic CUDA, and identical
within-seed evaluation seeds. The outcome-blind scale audit fixes

\[
 V=10^8,\qquad \beta=83650.1230871728,\qquad \nu=1.
\]

The equal-resource cap is 0.5 refresh per launch, or 2,048 refresh units.
The proposed controller uses a terminal hard cap so the dual queue can price
bursts. Strong baselines include exact cache-only Lyapunov, mismatch with
terminal and paced-prefix caps, an oldest-first complete burst, paced age,
round-robin and physical static-chain schedules, and no refresh. Signed-only
is a mechanism ablation. A rate-19 complete-refresh run is an explicitly
unequal-resource freshness ceiling, never a baseline for the main comparison.

## Frozen gates

All gates are mandatory:

1. all runs finite, drained, transition matched, and within their exact cap;
2. mean complete-refresh minus no-refresh terminal return at least 1.0 and
   positive in both seeds;
3. proposed refresh spending between 5% and 100% of the cap in both seeds;
4. proposed mean terminal return exceeds the best strong equal-resource
   comparator by at least 0.5 and in both seeds;
5. this gain recovers at least 10% of the complete/no-refresh headroom;
6. proposed mean terminal return exceeds both cache-only and signed-only by
   at least 0.25;
7. proposed mean return AUC is no worse than the best strong comparator;
8. proposed runtime is at most twice no-refresh runtime.

The strongest baseline is selected across both development seeds, which makes
the gate deliberately conservative. Any failure stops this controller
configuration before a pilot; thresholds, seeds, methods and optimizer
settings cannot be changed after outcomes. A passed matrix authorizes only a
separately preregistered pilot with new seeds and frozen hashes.
