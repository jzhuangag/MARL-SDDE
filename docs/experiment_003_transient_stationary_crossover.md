# Experiment 003: Transient-to-stationary participation crossover

## Material Passport

- Artifact type: exploratory mechanism analysis
- Experiment ID: `EXP-003-transient-stationary-crossover`
- Parent experiment: `EXP-001-dependence-delay-linear`
- Status: completed
- Registration: post-hoc; not a replacement for the failed EXP-001 gate
- Machine-readable outputs:
  `experiments/dependence_delay_linear/results/crossover/`

## Motivation

EXP-001 found strong correlation-limited speedup but did not find an interior
agent count after jointly tuning the step size for 500 iterations. Parameter
inspection suggested a transient-versus-steady-state tradeoff: additional
delayed agents improved contraction while raising the constant-step error
floor.

## Controlled analysis

The analysis fixes

\[
\eta=0.02,\qquad \rho=0.9,
\]

uses the registered heterogeneous delay profile with maximum delay 16, and
evaluates

\[
K\in\{25,50,100,200,500,1000,2000\}.
\]

Both sample-time and server-time common-factor alignments are retained. All
MSE values are exact augmented-system calculations; no curve is selected from
Monte Carlo noise.

## Results

- At \(K=25\), all 32 agents minimized MSE under both alignments.
- At \(K=50\), the optimum was 32 agents for sample-time alignment and 16 for
  server-time alignment.
- From \(K=100\) onward, 4 agents minimized MSE under both alignments.
- Under server-time alignment, the stationary MSE was \(0.25479\) at the
  4-agent optimum and \(0.28390\) with all 32 agents. Thus accepting every
  agent raised the fixed-step error floor by \(11.42\%\).

## Interpretation

The supported phenomenon is narrower and more defensible than “more agents
always hurt”:

> With persistent common Markov noise, heterogeneous staleness, and a fixed
> constant step, the best participation level can decrease as training moves
> from transient contraction to its stationary regime.

This suggests a low-complexity stagewise controller that adapts participation
as well as step size. It does not yet establish that such a controller beats a
jointly tuned fixed policy, nor that the crossover persists for temporal-
difference learning or nonlinear reinforcement learning.

## Next required test

Implement a predictable stagewise rule that estimates only scalar aggregate
noise and delay statistics from the previous stage. Compare it with:

- all agents with a tuned constant step;
- delay-only step adaptation;
- a joint oracle over step size and participation;
- a schedule that changes participation but ignores dependence.

The controller should be considered promising only if it tracks the oracle
crossover on unseen correlation and delay settings without estimating a full
cross-agent covariance matrix.
