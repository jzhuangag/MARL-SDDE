# T-051A fingerprint-controller static preregistration

## Purpose

T-051A is an outcome-free analytic gate for the T-051 fingerprint probe and
the T-050 stationary fixed-participation phase. It uses the unchanged public
Gymnasium kernels and does not read any T-049A outcome row.

The learning catalogue is (q\in\{1,4,16\}). For each task and delay, the
post-probe message budget is chosen so that the most expensive action,
(q=16), receives the public half-horizon contraction count required to make
the deterministic term at most (10^{-4}). This rule is strictly more
conservative than the (10^{-3}) diagnostic reported in T-050.

## Fully charged probe

The controller uses 96 independent blocks and two probe actors. Each actor
sends one SHA-256 state-path fingerprint. The fingerprint length is the
shortest task-specific length whose exact independent-path collision
probability is at most 0.01. Every block costs (h+2) message units and its
full path length in environment transitions. All learning resources and the
registered delay are charged separately.

The plug-in expected coefficient is bounded by T-051 Theorem 3. The primary
ratio compares its post-probe learning risk with the overhead-specific strong
fixed-(q) policy using the entire no-probe total message budget.

## Frozen stop rule

S1--S12 in the JSON plan are mandatory. Any failure stops this line before a
sampled CPU pilot. In particular, aggregate value alone cannot rescue a
failure of per-overhead value, active-cell breadth, 5% no-harm, exact budget
accounting, or reproducibility. No threshold, task, delay, correlation, or
probe size may be changed after execution under this identifier.

Passing T-051A would authorize only a separately preregistered local CPU
sampled pilot. It would not authorize formal seeds, GPU, HPC4, or a nonlinear
benchmark.
