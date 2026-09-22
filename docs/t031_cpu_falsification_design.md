# T-031 CPU falsification design

## Purpose

The first experiment is a kill test, not a result-generating pilot.  It must
determine whether identity-aware diversity has enough full finite-time
learning value to justify a new algorithm.  No GPU is needed.

## Layer 0: exact linear-Gaussian delayed SA

Use actor pools of 16, 32, and 64 agents with preregistered participation
counts in `{2,4,8,16}` and certified dependency groups.  All agents target the
same scalar or low-dimensional strongly monotone operator.  Their innovations
are marginally identical within each declared quality class and arise from
stationary AR Markov common, group, and private factors.  Delay profiles are
permuted by a frozen Latin-square map so that agent indices cannot encode the
answer.

For every fixed subset/template, propagate the complete augmented linear
state covariance.  The state includes the parameter delay ring and the
required Markov factor histories.  The objective is finite-horizon parameter
MSE AUC, not innovation variance divided by a horizon.

The solver must include:

- initial bias and contraction;
- long-run common/group/private covariance;
- identity-specific stale parameters and delayed innovations;
- message, total environment-interaction, and simulated wall-clock budgets;
- all costs of probing or profile certification.

If this recursion cannot be implemented exactly for Layer 0, the scan stops.

## Candidate policies

1. all-agent;
2. strongest count-only fixed-q rule, including `q=1`;
3. uniform random subset at the selected q;
4. quality-only;
5. freshness-only;
6. diversity-only;
7. strongest fixed subset per budget ray;
8. FDG certified block selector;
9. exhaustive or dynamic-programming template oracle, ceiling only.

## Frozen static gates for the future preregistration

- exact finite values and zero budget violations;
- oracle gain at least 15% relative to the strongest deployable fixed-subset
  baseline in aggregate full-risk AUC;
- FDG gain at least 12% and at least 80% of the oracle gain;
- at least 70% of heterogeneity-active cells improve by at least 5%;
- homogeneous controls within 2% of the strongest baseline;
- at least three distinct optimal templates and at least two distinct q
  values;
- quality-only, freshness-only, and diversity-only must each be worse than
  D3S on at least one preregistered conflict regime;
- at least one improvement path under each resource ray;
- old EXP-017A--019A outcomes and seeds are forbidden inputs;
- byte-identical rerun and complete unit tests.

The grid, thresholds, and implementation hashes must be committed before the
scan runs.  A failed mandatory gate prohibits an actual-learning pilot.

## Layer 1: exact-value tabular TD

Only after Layer 0 passes, preregister a 64-seed CPU pilot on Blackjack,
slippery FrozenLake, and Taxi or a second exact-value finite MDP.  Every agent
follows the same fixed policy
and target operator.  Dependence is induced by a documented shared exogenous
random source or an explicitly labeled zero-mean measurement channel while
preserving each marginal law.  The runner uses a real event queue: gradients
are bound to the parameter at generation time and applied on arrival.

The actual-learning gate is at least 10% aggregate normalized-MSVE AUC gain
over the strongest task-by-budget fixed subset, at least 5% on two individual
tasks, at least 5% terminal gain, at least 3% CVaR90 gain, at least 70%
active-cell directionality, inactive no-harm ratio at most 1.02, and controller
overhead below 5%.  A result below 5% aggregate stops the ICML line regardless
of the Layer-0 ceiling.

## Layer 2: natural correlation benchmark

At least one CPU or GPU benchmark must contain natural common shocks, such as
shared demand/weather/process disturbances across parallel control or queueing
instances.  An entirely injected common-factor evaluation is insufficient.

## Layer 3: nonlinear transfer

Two nonlinear suites, new seeds, and a separate preregistration are required
only after Layers 0--2 pass.  Suitable candidates are MinAtar and a continuous
control suite with vectorized actors.  Compare D3S to DASA/delay-only,
DELTA/cluster-style diversity baselines, all-agent, random subset, best fixed
q, and best fixed subset.

GPU is authorized only at Layer 3.  Current required compute is local CPU.
