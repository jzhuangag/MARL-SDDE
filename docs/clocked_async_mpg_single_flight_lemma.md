# Single-flight self-freshness lemma

Status: proved structural refinement for the executable one-worker-per-policy-
block architecture.

## Architecture

Each agent has at most one in-flight gradient packet.  Immediately after
agent `i`'s packet is applied, that worker reads the resulting joint policy,
starts its next fixed-length rollout and does not start another until the
packet completes.  Only agent `i` updates block `theta_i`.

If the packet arriving at event `k` was born at `b_(k,i)`, then no update of
block `i` occurs strictly between birth and completion.  Therefore

\[
\theta_i^k=\theta_i^{b_{k,i}}. \tag{1}
\]

The stale-gradient mismatch is consequently

\[
\|\nabla_i f(\theta^k)-\nabla_i f(\theta^{b_{k,i}})\|
\le\sum_{j\ne i}L_{ij}
\|\theta_j^k-\theta_j^{b_{k,i}}\|. \tag{2}
\]

The diagonal block smoothness `L_ii` remains in the one-step objective
remainder.  It is removed only from the **staleness history**, not from update
stability.

## Refined history and local steps

Let `L_off` equal `L` with zero diagonal and define

\[
\ell_i^{\rm off}=\sum_{j\ne i}L_{ij},
\qquad
\widetilde w_j=\sum_i p_i\alpha_i
\ell_i^{\rm off}L_{ij}^{\rm off}. \tag{3}
\]

The heterogeneous-step Lyapunov proof remains unchanged after replacing its
history weights by (3).  A valid sufficient condition is

\[
L_{ii}\alpha_i+(1+\delta)D^2\widetilde w_i\alpha_i\le1. \tag{4}
\]

For the low-complexity local-curvature family `alpha_i=s/L_ii`, form

\[
v_j=\sum_i p_i\frac{\ell_i^{\rm off}}{L_{ii}}L_{ij}^{\rm off}.
\]

Then (4) is

\[
s+(1+\delta)D^2\frac{v_i}{L_{ii}}s^2\le1. \tag{5}
\]

The maximal common scale is the minimum of `n` explicit positive quadratic
roots, and the complete calculation costs `O(n^2)`.  When cross-agent
sensitivity is zero, `v=0`, `s=1` for every delay: a packet cannot become stale
merely because unrelated teammates completed work.  This is the correct
distinct-agent limit and is lost in a generic delayed-coordinate bound that
charges diagonal sensitivity.

## Why the correction was necessary

The first tainted development grid used the generic history containing
`L_ii`.  It made the worst-case service-delay window shrink every local step
even at zero cross coupling, and the asynchronous method was slower than the
shadow barrier in all 16 development cells.  Those ratios are retained as
design-tainted diagnostics and never become evidence.

After proving (1)--(5), the implementation was changed before confirmation
freeze.  A second tainted single-seed grid showed the predicted qualitative
phase: all 12 cells with service ratio at least two improved over the fully
utilized shadow barrier, while homogeneous service was near parity and could
be worse under stronger coupling.  These values only justified freezing an
independent confirmation; they did not set formal outcomes.

## Boundary

The lemma fails with multiple simultaneous packets for the same policy block,
local actor updates outside the parameter server, or speculative same-agent
rollouts born before the previous packet is applied.  Such systems must restore
diagonal staleness or prove a different dependency graph.  The current CPU
confirmation and prospective theorem use exactly one in-flight packet per
agent.

The implementation verifies the refined drift on random coupled quadratics
whose histories satisfy (1), and verifies zero delay price on diagonal
objectives even for an arbitrarily large registered event window.
