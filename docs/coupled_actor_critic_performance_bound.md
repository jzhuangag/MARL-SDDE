# Conditional performance bound for the coupled actor--critic action

Date: 2026-09-05.

Status: **abstract conditional theorem proved; Markov-packet construction and
full-game conversion remain open.**  This document is not a claim that the
performance-bound kill gate has passed.

## 1. Executable certificate state

At event `k`, owner `i=I_k` returns a policy-gradient packet `g_hat_k` with
observable norm `g_k`.  The server stores its policy-version path and a scalar
critic certificate `c_k`.  On a simultaneous all-prefix confidence event,
assume

\[
 \|\nabla_i f(\theta_k)-\widehat g_k\|\le r_k,          \tag{1}
\]

where `r_k` is the sum of a declared Markov/statistical radius, critic-bias
radius and the off-diagonal version-path radius

\[
 d_k=\sum_{j\ne i}L_{ij}
       \|\theta_j^k-\theta_j^{b_k}\|.                  \tag{2}
\]

No additional sensing is needed for (2): the learner already owns the joint
policy versions.  The constants `L_ij` still require a theorem-facing bound;
an outcome-fitted neural proxy is not a certificate.

Suppose a critic correction of scale `beta` satisfies, on the same event,

\[
 \|e_k-\beta\widehat F_k\|
 \le(1-\mu_c\beta)c_k+\beta\epsilon_{c,k},
 \quad 0\le\beta\le L_c^{-1},                          \tag{3}
\]

and the critic fixed point moves by at most
`kappa_i ||theta_i^{k+1}-theta_i^k||`.  The predictable certificate update is

\[
 c_{k+1}=(1-\mu_c\beta_k)c_k+\beta_k\epsilon_{c,k}
          +\kappa_i\alpha_k g_k.                        \tag{4}
\]

Here the theorem-facing SPD critic operator satisfies
`mu_c I <= A_theta <= L_c I`.  The admissible step endpoint is governed by
`L_c`, not merely by `mu_c`.

This is a radius recursion, not a claim that TD loss equals value error.

## 2. Lyapunov action

Let `L_i` be block smoothness, and let `h_i>=0` denote any additional
new-step curvature supplied by a separately proved in-flight history shift.
Use the certificate Lyapunov function

\[
 \mathscr V_k=f(\theta_k)+\lambda c_k^2+\mathcal H_k.   \tag{5}
\]

Smoothness and (1) give

\[
 f(\theta_{k+1})-f(\theta_k)
 \le-\alpha_k g_k(g_k-r_k)
     +\tfrac{L_i}{2}\alpha_k^2g_k^2.                   \tag{6}
\]

If the history lemma contributes at most
`h_i alpha_k^2 g_k^2/2` after its negative shift, then (4)--(6) imply

\[
 \mathscr V_{k+1}-\mathscr V_k\le U_k(\alpha_k,\beta_k),\tag{7}
\]

where

\[
 U_k=-\alpha g(g-r)+\frac{L_i+h_i}{2}\alpha^2g^2
 +\lambda\{[c+\kappa_i\alpha g
 +(\epsilon_c-\mu_cc)\beta]^2-c^2\}.                  \tag{8}
\]

Equation (8) is a convex quadratic in `(alpha,beta)`: its Hessian is a
nonnegative actor-curvature diagonal plus a rank-one Gram matrix.  The online
action is its exact box minimizer.  It is computed by checking one interior
stationary point and four faces, so the control layer is constant size.

Because `(0,0)` is feasible, `U_k(alpha_k,beta_k)<=0`.  Thus (5) is
nonincreasing on the declared confidence event.  More importantly, Lyapunov
is the design objective that jointly selects both timescales.  `beta` is not
chosen offline and `alpha` is not a post-hoc safety clip.

## 3. Conditional finite-time implication

Define

\[
 C_i=L_i+h_i+2\lambda\kappa_i^2,
 \qquad
 R_k=r_k+2\lambda\kappa_i c_k.                          \tag{9}
\]

Compare the joint minimizer with the feasible action `beta=0` and the best
clipped actor scale.  Direct minimization of the resulting scalar quadratic
gives

\[
 -U_k(\alpha_k,\beta_k)
 \ge\frac12\min\{\bar\alpha_k,C_i^{-1}\}
       [g_k-R_k]_+^2.                                   \tag{10}
\]

Telescoping (7), using nonnegativity of (5), yields

\[
 \sum_{k=0}^{K-1}
 \min\{\bar\alpha_k,C_{I_k}^{-1}\}
 [g_k-R_k]_+^2
 \le2\mathscr V_0.                                     \tag{11}
\]

The proof of (10) has two cases.  If the unconstrained scalar minimizer lies
inside the actor box, its quadratic improvement is at least
`[g-R]_+^2/(2C_i)`.  If the cap binds, the improvement is at least
`bar_alpha [g-R]_+^2/2`.  The full two-action minimizer cannot be worse than
this comparator.

Similarly, the `alpha=0` comparator shows that when
`c_k>2 epsilon_c,k/mu_c`, a fixed feasible critic step produces a decrease
proportional to `lambda c_k^2`.  Hence the average critic radius approaches
its confidence floor rather than being assumed negligible.

Finally, (1) and (9) imply

\[
 \|\nabla_i f(\theta_k)\|
 \le[g_k-R_k]_+ +2r_k+2\lambda\kappa_i c_k.             \tag{12}
\]

Equations (11)--(12) give an activated-block stationarity bound with explicit
statistical, strategic-staleness and critic-tracking floors.  With a
`K^{-1/2}` actor cap, shrinking confidence radii and bounded service coverage,
the comparator term is `O(K^{-1/2})`; the existing block-coverage lemma adds
the declared coverage and intervening-motion terms needed for a full-gradient
average.

This is a real finite-time certificate theorem, but it is not yet the desired
Markov-game theorem.  In particular, (11) cannot silently replace `r_k` by an
empirical TD residual or assume that (2) vanishes.

## 4. What remains to close the performance-bound gate

Four items remain mandatory.

1. **Markov packet radius.**  For fixed-horizon reset trajectories, construct
   an all-prefix vector confidence radius around the actual actor packet.  A
   valid route treats each whole trajectory as one bounded vector and uses
   independent reset replicas; it does not treat within-trajectory transitions
   as iid.  Continuing-chain sampling needs a separate Poisson-equation or
   mixing argument.
2. **Critic contraction radius.**  Establish (3) for the exact linear/tabular
   critic update under the packet's behavior policy, including policy-version
   mismatch.  A sample split may be used, but every trajectory and critic
   update must be charged.
3. **History closure.**  The first executable theorem sets `h_i=0` and pays
   policy and critic staleness exactly once in the robust radii (1)--(3).  A
   Lyapunov--Krasovskii refinement is optional and may only replace, rather
   than duplicate, the same version-path term.
4. **Game conversion.**  Combine activated-block coverage with a declared
   Markov-potential-game stationarity/Nash conversion and state every
   occupancy/interiority constant.  General-sum or neural guarantees are not
   inherited.

Only after all four are proved for one executable packet format can the
performance-bound gate pass.  The exact oracle-headroom result does not lower
this standard.
