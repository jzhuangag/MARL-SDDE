# T-049A preregistration: exact standard-task schedule-value scan

## Question

T-049A asks whether a prospectively frozen library containing fixed and
two-stage participation schedules has enough exact, full-cost value on three
public Markov tasks to justify building a learned controller. It is a
deterministic CPU feasibility scan, not a pilot or formal experiment.

The scan differs from T-045A in two substantive ways fixed before execution.
First, it uses the complete vector temporal-difference drift and every
matrix-valued gradient lag covariance from T-049, rather than a scalar
minimum-drift/noise-trace/second-eigenvalue proxy. Second, its
marginal-preserving trajectory-switch coupling leaves every actor's
Gymnasium trajectory law unchanged while producing the exact T-047
time-varying prefix-overlap covariance.

## Frozen standard-task population

The task set is:

- slippery 8-by-8 FrozenLake-v1 with eight Fourier features;
- CliffWalking-v0 with twelve Fourier features;
- Taxi-v3 with sixteen Fourier features.

Gymnasium is pinned to version 1.0.0. Each task uses exact value iteration
with the smallest-action tie break, followed by a 0.1 epsilon-soft mixture.
The discount is 0.95. Terminal transitions use zero bootstrap and then reset
from the exact upstream initial distribution. No task may be removed after
the scan.

## Frozen schedules and budgets

The fixed catalogue is \(q\in\{1,4,16\}\). The nonconstant catalogue changes
between \(q=1\) and \(q=16\) after 25%, 50%, or 75% of the maximal feasible
horizon, in both directions. Every schedule maximizes its update count under
the same message and environment budgets.

The message overhead is \(h\in\{8,32\}\). For reference horizons
\(H\in\{96,192\}\), the message budget is \(H(h+4)\). Delays are
\(D\in\{0,4,8\}\), and correlations are
\(\rho\in\{0,.1,.3,.5,.7,.9,1\}\). These values follow the asymptotic phase
\(q^\star=\sqrt{h(1-\rho)/\rho}\): they span interior and boundary optima
without using any task risk outcome.

The estimator averages the final half of the iterates and uses

\[
 \eta=0.75(1-\lambda)/\lVert A\rVert_{\mathrm{op}},
\]

where \(\lambda\) and \(A\) are exact public task constants.

## Fully charged optimistic probe ceiling

The post-probe oracle pays for four independent restart blocks with
\(q_{\mathrm{probe}}=16\). Each block is advanced for the public mixing
burn-in required to make the second-largest-eigenvalue-modulus envelope at
most \(10^{-4}\). Probe messages, environment transitions, and one latency
charge are subtracted before learning, and probe gradients are not reused.

This is deliberately an optimistic value ceiling: it grants the oracle the
true correlation after paying the registered resources. It does not evaluate
or authorize a learned correlation estimator. If even this ceiling fails,
estimation cannot rescue the design.

## Analysis and gates

For each task, overhead, reference horizon, and delay, the strong fixed
baseline is the fixed \(q\) with minimum geometric exact full-budget risk over
the complete registered correlation grid. It therefore adapts to every
public system coordinate except the unknown correlation. The cellwise fixed
oracle is reported separately.

All ten gates V1--V10 in the machine-readable configuration are mandatory.
In particular:

1. the fully charged schedule oracle must improve the strong fixed baseline
   by at least 5% geometrically and strictly in at least 50% of cells;
2. the best nonconstant post-probe schedule must improve the cellwise
   full-budget fixed oracle by at least 1% geometrically and strictly in at
   least 25% of cells;
3. mean oracle participation must be nonincreasing in at least 80% of
   adjacent-correlation comparisons;
4. the selected support must include at least two fixed schedules and one
   nonconstant schedule.

Any failure forbids a learned EXP-021A pilot. Tasks, thresholds, schedules,
costs, and populations may not be changed after execution. A pass would
authorize only a separate probe-feasibility theorem and preregistration; it
would not authorize sampled trajectories, formal seeds, GPU, or HPC4.

## Frozen workload and provenance

The grid contains 36 base scenarios, 252 correlation cells, nine schedules
per cell, and 2,268 deterministic rows. The maximum full learning horizon is
256 updates. The prescribed hardware is local CPU.

The preregistration is frozen by its Git commit. The runner refuses to
overwrite an existing output directory and records SHA-256 hashes for the
configuration, runner, both theorem cores, task constants, rows, and summary.
