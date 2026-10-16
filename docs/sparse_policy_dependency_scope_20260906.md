# Sparse policy-dependency scope for controlled asynchronous MARL

Date: 2026-09-06

Status: theorem and estimator scope.  This is a material estimator redesign
after PDSG-COMP-001; it does not alter that failed timing gate and does not
authorize a successor outcome experiment.

## 1. What is sparse

The paper does not assume that a sparse execution communication graph is given.
Instead, for each owner block `i`, fix a prospective policy-dependency
neighborhood `N_i` before evaluating learning outcomes.  The dynamic refresh
graph may activate edges only from this support:

\[
S_{i,k}\subseteq\{j\to i:j\in\mathcal N_i\}.
\tag{1}
\]

The support says which teammate policy versions can materially change the
owner's policy-gradient mean.  The active edge set says which of those versions
is transmitted at a particular asynchronous dispatch.  The support is static;
the scheduling graph is dynamic.

## 2. True-gradient compressibility assumption

Let `G_i(theta_i,x_-i)` denote the population owner-gradient mean when the
owner is current and teammate policies are `x_-i`.  Assume the block envelope

\[
\|G_i(\theta_i,x_{-i})-G_i(\theta_i,y_{-i})\|
\le\sum_{j\ne i}L_{ij}\|x_j-y_j\|.
\tag{2}
\]

For the registered trust region and cache states, define the omitted tail

\[
\varepsilon_{i,k}^{\rm sp}
=\sum_{j\notin\mathcal N_i}L_{ij}
\|\theta_j^k-\chi_{j\to i}^k\|.
\tag{3}
\]

The theorem may use either exact sparsity, where (3) is zero, or approximate
sparsity with a declared bound.  Reward locality or a factorized neural critic
alone does not prove (3): globally coupled dynamics can make a distant policy
affect future occupancy and hence the owner gradient.  The true-gradient tail
must be assumed or separately certified.

This point prevents a common but invalid argument that a local reward graph
automatically implies a local policy-gradient graph.

## 3. Factorized critic implementation

An executable centralized surrogate may be written

\[
\widehat F(\theta;\mathcal B)
=\sum_{f\in\mathcal F}\widehat F_f
(\theta_{S_f};\mathcal B_f).
\tag{4}
\]

The owner-gradient surrogate depends only on

\[
\widehat{\mathcal N}_i
=\bigcup_{f:i\in S_f}(S_f\setminus\{i\}).
\tag{5}
\]

For `j` outside (5), the surrogate cross-Hessian block is exactly zero.  One
reverse derivative of the owner-gradient alignment scalar therefore touches
only the owner factors and the donor parameters in (5).  With
`Delta=max_i|N_i|`, the scheduling computation and retained derivative blocks
scale as `O(Delta)` times the local actor/factor cost, not with the total number
of agents.

Equation (4) is an implementation property.  Agreement between (5) and the
true dependency tail in (3) is a statistical approximation obligation.

## 4. Explicit sparse-approximation price

Let `g_i` be the true candidate gradient and `g_i^loc` its local surrogate,
with

\[
\|g_i-g_i^{\rm loc}\|\le\varepsilon_i^{\rm sp}.
\tag{6}
\]

For current objective gradient `a_i`, the alignment error obeys

\[
|\langle a_i,g_i\rangle-
\langle a_i,g_i^{\rm loc}\rangle|
\le\|a_i\|\varepsilon_i^{\rm sp},
\tag{7}
\]

and

\[
|\|g_i\|^2-\|g_i^{\rm loc}\|^2|
\le\varepsilon_i^{\rm sp}
(2\|g_i^{\rm loc}\|+\varepsilon_i^{\rm sp}).
\tag{8}
\]

Consequently, for `0 <= alpha <= alpha_max`, the additional smooth drift-score
error is at most

\[
r_{i,k}^{\rm sp}
=\bar\alpha\|a_i\|\varepsilon_i^{\rm sp}
+\frac{L_i\bar\alpha^2}{2}
\varepsilon_i^{\rm sp}
(2\|g_i^{\rm loc}\|+\varepsilon_i^{\rm sp}).
\tag{9}
\]

This term is added to the expectation-level selection error.  Sparsity cannot
be treated as free: if (9) is comparable to the dynamic oracle margin, the
low-degree controller has no theorem-facing adaptation advantage.

## 5. Dense-game behavior

For dense or noncompressible games, the method has three honest outcomes:

1. use the dense neighborhood and pay linear-in-agent derivative cost;
2. use a sparse approximation and report the explicit floor (9);
3. fall back to a registered static synchronization schedule.

It does not claim degree-independent computation or universal improvement in
dense games.  Dense environments remain controls for the approximation floor
and fallback behavior.

## 6. Next mandatory gates

Before another timing or learning run:

1. mechanically verify that a factorized surrogate produces zero derivative
   blocks outside (5);
2. validate (7)--(9) against exact quadratic objectives;
3. freeze a new local-factor cost audit whose implementation never forwards or
   differentiates non-neighbor actors in the scheduling calculation;
4. measure scaling at fixed `Delta` as total agent count grows and at increasing
   `Delta` under a linear, not degree-independent, claim;
5. design a separate outcome-free tail audit for candidate standard MARL
   benchmarks.

Only the first four are local engineering/theory tests.  Gate 5 determines
whether the sparse scope transfers beyond the exact development model.
