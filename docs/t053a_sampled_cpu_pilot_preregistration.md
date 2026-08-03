# T-053A sampled standard-task CPU pilot preregistration

## Scope

T-053A is the first sampled test of the T-052 controller. It uses exact
policy-weighted Gymnasium action/transition/reward outcomes rather than a
Gaussian proxy or an edge-conditional mean. The sampled kernel is checked
against the public regenerative transition, innovation mean, and full
innovation second moment before any endpoint is accepted.

The pilot has 84 cells and eight new seeds, yielding 672 paired endpoints.
Each endpoint creates one common and 16 private stationary learning
trajectories. The same trajectory bank and prefix actors are reused by the
controller and every fixed-(q) comparator. Probe paths use an independent
random stream.

## Budgets and estimators

For each task and delay, the post-probe message budget ensures that (q=16)
receives the public (10^{-4}) contraction horizon. Ninety-six two-agent
fingerprint messages and every fingerprint transition are charged before
learning. The strong fixed and oracle comparators receive the complete
no-probe total message budget.

The sampled recursion is the exact additive delayed projected-TD model used
by the theory. The endpoint is the squared Euclidean norm of its half-tail
Polyak--Ruppert average. This pilot does not claim a multiplicative nonlinear
TD theorem.

## Frozen stop rule

P1--P12 are mandatory. A failed task, delay, active-cell breadth, inactive
no-harm, probe calibration, budget, or reproducibility gate stops formal
execution. Pilot seeds cannot become formal seeds. Passing authorizes only a
separate formal preregistration; it does not itself authorize GPU, HPC4, or a
nonlinear benchmark.
