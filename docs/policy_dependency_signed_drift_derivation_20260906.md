# Signed drift minimization for one-edge policy refresh

Date: 2026-09-06

Status: exact algebra and algorithm interface.  No new scientific trajectory
or empirical outcome is introduced.

## 1. Why unsigned mismatch is insufficient

Let the current expected quadratic potential be

\[
F(\theta)=\frac12 e^\top\bar H e,
\qquad e=\theta-\theta^*,
\]

and let owner `i` receive a state-`s` gradient computed from its teammate
cache.  Before a new packet is launched, refreshing donor `j` changes the
packet's noiseless scalar gradient from `g_0` to

\[
g_j=g_0+d_{ij},
\qquad
d_{ij}=H_{s,ij}(\theta_j^{k-d}-\chi_{j\to i}).
\]

The current true expected-potential block gradient is

\[
a_i=e_i^\top\bar H e.
\]

For a fixed step `alpha`, the exact expected-potential increment caused by
candidate `j` is

\[
\Delta_F(j,\alpha)
=-\alpha a_i g_j
+\frac12\alpha^2\bar H_{ii}g_j^2.
\tag{1}
\]

Equation (1) follows by expanding the quadratic after changing only coordinate
`i`.  The sign of `a_i g_j` decides whether the candidate is locally
descending.  Neither `|H_(s,ij)|` nor cache-mismatch magnitude contains that
sign.

## 2. Lyapunov edge value

For the composite Lyapunov function in
`controlled_sdde_sync_graph_mainline_20260906.md`, add the cache and queue
increments.  If the refresh delivers the delayed source
`theta_j^(k-d)` rather than the current `theta_j^k`, its exact cache-debt
reduction is

\[
R_{ij,k}=\frac{p_{ij}}2\left(
\|\theta_j^k-\chi_{j\to i}^k\|^2
-\|\theta_j^k-\theta_j^{k-d}\|^2
\right).
\tag{2}
\]

The second term is the residual flight mismatch.  Consequently a delayed
refresh can have negative immediate cache value.  Omitting this term is valid
only for zero delivery delay.  The candidate score is

\[
\Psi_j(\alpha)=
-\alpha a_i g_j
+\frac12\alpha^2\bar H_{ii}g_j^2
-R_{ij,k}
+Q_k c_{ij}.
\tag{3}
\]

The no-refresh candidate has `d_i0=0`, no reset benefit, and zero message
price.  With at most one edge per launch, minimizing (2) over all incoming
neighbors is exact and costs one scalar score per edge.

For each candidate, joint step-size minimization has the closed form

\[
\alpha_j^*=
\Pi_{[0,\alpha_{\max}]}
\frac{[a_i g_j]_+}{\bar H_{ii}g_j^2},
\tag{4}
\]

with `alpha_j^*=0` when `g_j=0`.  Substitution into (2), followed by an
`argmin` over `j`, jointly selects the refresh edge and step without a generic
QP, Hessian inverse, or finite hyperparameter scan.  Complexity is
`O(deg(i))` once candidate gradient changes are available.

## 3. Nonquadratic smooth potential

For an `L_i`-smooth potential, replace equality (1) by

\[
F(\theta-\alpha U_i g_j)-F(\theta)
\le
-\alpha\langle\nabla_iF(\theta),g_j\rangle
+\frac{L_i\alpha^2}{2}\|g_j\|^2.
\tag{5}
\]

An executable deep-MARL rule needs predictable estimates of the two quantities
in (4).  One feasible interface uses the previous completed trajectory block
to choose the next launch graph:

- `hat a_i`: the centralized critic's current actor-gradient estimate;
- `hat d_ij`: a cross-block Jacobian-vector product applied to the known policy
  version displacement;
- `epsilon_ij`: a simultaneous confidence bound for the prediction error;
- `L_i`: a public or certified block-smoothness envelope.

The scalar first-order part for all teammate blocks can be approximated from
one reverse-mode derivative of a gradient-alignment scalar and then
partitioned by policy block.  This proposed interface still needs a measured
autodiff-cost audit; the current evidence does not establish that it always
costs exactly one ordinary backward pass.  A certified norm bound must pay the
unobserved Taylor remainder.

The launch graph must be committed before the next trajectory is observed.
Using the same packet to choose its own refresh edge would be post-treatment
selection and is forbidden.

The frozen PDSG-001 `one_step_oracle` uses the registered target, expected
quadratic matrix, current Markov state, and candidate cache.  It is an
optimistic problem-value ceiling, not the executable estimator described
above.  Its positive result cannot be promoted to an algorithm result.

## 4. What remains to prove

Equations (1)--(3) are exact for the registered quadratic model.  They do not
yet justify the deep rule.  A theorem freeze requires:

1. a martingale/Poisson decomposition for `hat a_i` and `hat d_ij` under the
   mixed-version Markov trajectory law;
2. a uniform confidence event over all eligible edges and launch times;
3. a delayed Lyapunov--Krasovskii recursion that aligns launch-time graph
   choice with receipt-time policy progress;
4. mean-rate stability of the communication queue;
5. a nonvacuous drift bound after the confidence penalty;
6. an autodiff complexity measurement showing the signal is affordable;
7. a finite-horizon controlled-hybrid-SDDE approximation theorem, if the SDDE
   is retained in the paper.

If the confidence penalty always selects no refresh, or if the extra gradient
calculation costs more than the oracle gain, the practical architecture fails
despite the positive PDSG-001 problem signal.
