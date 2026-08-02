# T-030 W0 diagonal weighted-MSVE audit preregistration

## Scope

The Euclidean Theorem 4 certificate proved at most 0.0105612% reduction on
Blackjack even under impossible best-case assumptions.  W0 tests the only
predeclared low-memory weighted extension before any further proof or sampled
experiment.

The frozen metric family is

\[
W_\theta=\operatorname{diag}(\pi^\theta),\qquad
\theta\in\{0,1/4,1/2,3/4,1\}.
\]

It is fixed independently of EXP-019A outcomes.  Every metric is diagonal, so
the existing coordinate projection is nonexpansive in its norm and the proof
artifact uses O(d) metric storage.  The deployed TD update is unchanged;
`W_theta` is not a preconditioner.

Configuration SHA-256:
`84cc7fde97b1dea73cf23d579f55734fcb6237979451ff1b4cc8009e463d4ebf`.

## W0 optimistic gate

For each theta, transform the exact Blackjack mean Jacobian and same-time
second-Jacobian moment into the W norm.  Set total-variation mixing error,
innovation forcing, and delay to zero.  Minimize

\[
1-\eta\mu_W+2K_W\eta^2
\]

in closed form, and additionally combine the smallest curvature from any
registered `(q,rho)` action with the largest update count from any budget
cell.  This is an optimistic relaxation; the real certificate cannot be
better.

W0 passes if and only if at least one frozen metric can prove at least 5%
reduction.  If none passes, W1/W2, a new sampled CPU experiment, Asterix,
HPC4, and GPU all remain stopped.  No theta, gate, q/rho catalogue, task
constant, or budget is changed after this preregistration.
