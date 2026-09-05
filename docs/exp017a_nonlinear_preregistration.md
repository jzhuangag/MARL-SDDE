# EXP-017A nonlinear GPU pilot preregistration

## Outcome-free status

This commit freezes the standard-task nonlinear benchmark before any EXP-017A
trajectory exists. It does not modify EXP-016B code, seeds, gates, results, or
claims. Pilot seeds are implementation-only and permanently excluded from any
formal confirmation.

- Parent HEAD: `a5f25667217bea72cce55a9aa44a6b991f6847f9`
- Configuration SHA-256: `28c1c24181c6de02fd0b48e7f420c3ea46887024ccb0551ccef9597f109ae5ea`
- Static manifest SHA-256: `bf83f1031533205a7395338cae0bcff716692c95214ecaede8956f621bdfb75b`
- Pilot seeds: `[20550101, 20550102]`
- Formal seeds: **not assigned**
- Expected pilot endpoints: 1584

## Tasks and learner

The benchmark uses Gymnasium `CartPole-v1` and `Acrobot-v1` under frozen
stochastic behavior policies. The learner is a two-hidden-layer ReLU neural
TD(0) predictor trained by plain SGD. There is no actor--critic, Hessian or
covariance inverse, preconditioner, or target-policy adaptation.

The held-out metrics are terminal Monte-Carlo prediction MSE and empirical
mean-squared Bellman error. Normalized prediction-error AUC, communication,
environment steps, total agent transitions, CVaR90, wall time, controller
overhead, and the complete selected `(q,b)` path are also frozen.

## Dependence and mixing construction

For every agent, a complete source trajectory is selected from either one
common source (probability `sqrt(rho)`) or an iid private source. Common and
private sources have exactly the same task and regeneration law. Thus each
single-agent marginal is invariant in `rho`, while two agents share their
source with probability `rho`. No observation noise is added to manufacture an
advantage.

At public regeneration events all source environments reset from their
standard Gymnasium initial laws. This is a joint Doeblin minorization:
`lambda_upper <= 1-gamma`, with frozen pairs `(0.8,0.2)` and `(0.95,0.05)`.
The experiment supports only this known-mixing setting.

## Scenarios and baselines

The Cartesian grid contains two tasks, two mixing profiles, correlations
`[0,0.5,0.9]`, zero/edge-jitter/WAN-bursty delay traces, and message- versus
environment-binding budgets. Every paired policy receives the same byte and
environment budgets. Mandatory arms are oracle (evaluation only), always-all,
fixed q=4/16/32 (forming the best-fixed-q envelope), single-agent,
information-only, learning-aware, no-delay, correlation-blind, and
mixing-blind.

The nonzero delay traces are realistic-shaped deterministic synthetic traces,
not claimed to be measurements from a deployed network.

## Inference and progression

The pilot uses descriptive frozen gates only. A later formal analysis, if
authorized, uses one-sided paired seed-block sign-flip maxT inference with
100,000 frozen resamples and familywise alpha 0.05. Pilot outcomes may select
the fixed-q baseline for the later formal registry, but formal seeds must be
new and independently committed before use.

All 12 gates are mandatory. Any failure stops formal without
gate, seed, population, or threshold adjustment. Active outputs and the
scratch environment must remain under `/scratch/jzhuangag`; `/project` is not
needed for this pilot.
