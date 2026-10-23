# A theorem-facing cross-sensitivity interface for factorized policies

Status: finite-horizon analytic bound and exact finite-game validation.  The
bound is intentionally conservative.  It does not claim that a sampled PPO
ratio is a uniform neural certificate.

## From teammate logits to an owner-gradient bound

Fix owner agent `i` and change only teammate `j`'s policy.  Let the packet use
`H` transitions and let its pathwise Euclidean norm be at most `C_(i,H)`, as in
the fixed-horizon Markov-packet lemma.

For a categorical softmax with `A_j` actions, the Jacobian
`diag(p)-p p^T` has spectral norm at most `1/2`.  Therefore

\[
 \operatorname{TV}(\pi_{j,z},\pi_{j,z'})
 \le \frac{\sqrt{A_j}}4\|z-z'\|_2 .                 \tag{1}
\]

If the maximum statewise teammate-logit displacement is `Delta_z`, a coupling
of the two trajectory laws gives

\[
 \operatorname{TV}(P_{0:H},P'_{0:H})
 \le H\frac{\sqrt{A_j}}4\Delta_z .                  \tag{2}
\]

For a vector statistic bounded by `C_(i,H)`, changing its law changes its mean
by at most twice the statistic bound times total variation.  Hence

\[
 \|\nabla_iJ_{i,H}(\theta)-
   \nabla_iJ_{i,H}(\theta')\|
 \le
 \frac{H\sqrt{A_j}}2 C_{i,H}\Delta_z .             \tag{3}
\]

No transition inside a trajectory is treated as independent.  Equation (2)
is a path-law coupling bound, and (3) compares exact expectations.

If a policy network has a declared uniform parameter-to-logit constant

\[
 \max_s\|z_j(s;\vartheta)-z_j(s;\vartheta')\|_2
 \le K_j\|\vartheta-\vartheta'\|_2,
\]

then a valid theorem-facing cross coefficient is

\[
 L_{ij}^{\rm net}
 =\frac{H\sqrt{A_j}}2 C_{i,H}K_j.                  \tag{4}
\]

For tabular logits, `K_j<=1` under the full Euclidean parameter norm.  For a
neural actor, `K_j` can be upper-bounded from feature and spectral-norm bounds.
Such a worst-case product may be too large to give a useful step; that is a
nonvacuity question, not permission to replace it silently by an empirical
average.

## Relation to the main Lyapunov theorem

Equation (4) can populate the off-diagonal `L_ij` matrix in the
Lyapunov--Krasovskii history weights.  The same coefficient also bounds the
interaction uncertainty `B_k` in the lower phase certificate.  Thus the upper
and lower sides use one declared object.

The bound is fully observable for tabular softmax and for networks whose
parameter/logit Lipschitz envelope is recorded.  A practical implementation
may report a measured local proxy in addition, but the paper must keep the
following scopes separate:

- theorem: public uniform `L_ij^net` and the resulting certified step;
- implementation: any measured local cross-policy proxy;
- experiment: realized return, lag and utilization without pretending the
  local proxy proves uniform coverage.

Before GPU preregistration, a CPU static audit must instantiate (4) on the
actual actor architecture and determine whether its certified step is
nonzero and computationally meaningful.  Failure of that audit prevents a
claim that the neural implementation is theorem-covered; it does not alter the
tabular theorem.

## Validation

The implementation checks (1) on 1,200 random finite logit pairs.  It then
changes one teammate's logits in 20 random two-agent, two-state Markov games
and verifies that (3) covers the exact finite-horizon owner-gradient change
computed by dynamic programming.  These are deterministic bound checks, not a
learning experiment.
