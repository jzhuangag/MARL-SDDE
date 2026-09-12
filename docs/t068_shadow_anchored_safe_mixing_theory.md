# T-068 shadow-anchored safe personalized mixing: theorem-feasibility gate

## Decision and scope

The new mainline is mathematically feasible only after narrowing the claim to
fully observed, fixed-policy Markov learning with affine strongly monotone mean
operators.  It is not yet a theorem for nonlinear actor--critic, off-policy
bootstrapping, POMDPs, or cooperative Markov games.

Training is server-coordinated and federated.  Raw trajectories remain local.
Each agent deploys its own parameter without action-time communication.  The
server is neither a centralized critic nor a centralized trajectory learner.

The claim-critical idea is not generic graph discovery.  Each agent maintains
a collaborative model and a same-data local shadow model.  A recipient accepts
a stale donor mixture only when independent local Markov validation blocks
certify that the mixture is no worse than its *charged* shadow, which pauses
while validation data are collected.  The zero-transfer action is always
feasible after sensing.  Comparison with a no-probe local learner requires a
separate, explicit information-opportunity term; it is not free safety.

## Model class

For agent `i`, the projected fixed-policy TD mean field is

\[
F_i(\theta)=A_i\theta-b_i=A_i(\theta-\theta_i^\star),
\]

where `sym(A_i) >= mu_i I`, `||A_i|| <= L_i`, features and rewards are bounded,
and the behavior chain is geometrically beta-mixing.  The theorem-facing risk
is

\[
L_i(\theta)=\frac12\|F_i(\theta)\|^2.
\]

Nonsingularity gives

\[
\frac{\sigma_{\min}(A_i)^2}{2}\|\theta-\theta_i^\star\|^2
\le L_i(\theta)
\le
\frac{\|A_i\|^2}{2}\|\theta-\theta_i^\star\|^2.
\]

Thus the residual Lyapunov function is equivalent to personalized parameter
error on the registered model class.

## Shadow-anchored decision

Agent `i` updates a shadow `theta_i^L` using only its local TD samples.  At a
predictable communication checkpoint, define the candidate directions

\[
D_{i,t}=
[\theta_{i,t}-\theta_{i,t}^L,
\theta_{j_1,t-\tau_1}-\theta_{i,t}^L,\ldots].
\]

For weights in `W_i={w>=0: 1^T w<=1}`, the candidate is

\[
\theta_i(w)=\theta_{i,t}^L+D_{i,t}w.
\]

The exact Lyapunov change relative to the local shadow is the convex quadratic

\[
L_i(\theta_i(w))-L_i(\theta_{i,t}^L)
=g_{i,t}^T w+\frac12w^T H_{i,t}w,
\]

where

\[
g_{i,t}=(A_iD_{i,t})^TF_i(\theta_{i,t}^L),
\qquad
H_{i,t}=(A_iD_{i,t})^T(A_iD_{i,t})\succeq0.
\]

This identity is exact and removes the need to interpret donor gradients as
recipient gradients.

## Observable certificate and online optimization

Two disjoint post-gap local Markov blocks estimate `g_i` and `H_i`.  Candidate
directions are frozen before those blocks.  For finite horizon `T`, geometric
beta-mixing concentration plus a union bound across agents, checkpoints, and
quadratic coefficients yields a simultaneous radius `r_i,t(delta)`.  The gap
contributes an explicit mixing-bias term and every validation transition is
deducted from the learning budget.

The controller solves the recipient-side convex program

\[
\min_{w\in W_i}
\widehat g_i^Tw+\frac12w^T\widehat H_i^+w
+Q_t^{\rm comm}c_i^Tw
\]

subject to

\[
\widehat\Delta_i(w)+r_{i,t}(\delta,w)\le0.
\]

Here `H_hat_i^+` is a confidence-adjusted positive-semidefinite majorizer.
When the constraint has no nonzero feasible weight, `w=0` returns the exact
local shadow.  For at most `K` public candidate donors, online work is
`O(L K d + K^3)` per decision; fixed small `K` makes it linear in parameter
dimension.  No `d x d` covariance, Hessian inverse, or mixed-integer solver is
required.

## Proof obligations that close on this model class

1. **Quadratic identity.**  The displayed `g,H` expansion is algebraic and
   exact.
2. **Uniform observable bound.**  Block concentration controls every entry of
   `g_hat-g` and `H_hat-H`; simplex norm bounds make the result uniform over the
   continuum of feasible `w`.
3. **Anytime transfer safety.**  On the simultaneous confidence event, every
   accepted checkpoint satisfies
   `L_i(theta_i,t) <= L_i(theta_i,t^L)` relative to the charged local shadow.
   The fallback is an actual parallel local trajectory, so the comparator does
   not change with the collaborative state.
4. **Finite-time risk.**  Strong monotonicity and the Poisson-equation
   decomposition give the local-shadow Markov TD rate.  Combining it with the
   checkpoint certificate yields the same bound plus the registered confidence
   slack for every agent.
5. **Oracle excess risk.**  On a separated class where useful and harmful
   mixtures are at least `Delta` from the safety boundary, confidence widths
   identify the correct side after an explicit charged sample threshold.  A
   uniform matching result without separation is neither required nor true.
6. **Delay.**  The certificate evaluates the actually received stale model, so
   delay cannot invalidate safety.  Under bounded parameter drift, comparison
   with a fresh-model oracle incurs an additive term proportional to the delay
   bound.  Delay is a training-time communication effect, not an execution-time
   observation.
7. **Information cost.**  Relative to a no-probe local learner, the guarantee
   contains an explicit opportunity term `C_probe`.  A stronger end-to-end
   no-harm statement is available only when the certified transfer decrease is
   at least `C_probe`.  This qualification is necessary: unknown usefulness
   cannot be identified at zero cost.
8. **Budgets.**  A virtual communication queue prices average usage, while a
   residual-budget shield enforces exact pathwise feasibility.  Actor-transition
   costs include every learning, gap, and validation transition.

These obligations are substantial but mutually compatible.  The statement is
therefore proof-feasible, not already proved.  Exact no-harm against a no-probe
local learner on every unknown instance is deliberately not claimed.  The first falsification step is
an exact-moment CPU phase scan.  It must show that an endogenous
collaborate-then-personalize schedule has material value over local-only, full
sharing, and the best fixed mixing strength before a sampled controller is
preregistered.

## Role of Lyapunov and SDDE

`L_i` is the learning Lyapunov function and the transfer program minimizes its
certified drift.  Communication queues add resource shadow prices.  An SDDE is
not needed for correctness because stale candidates are tested directly on the
recipient.  A stochastic-delay continuous-time limit may be derived later as
an interpretation, but it is excluded from the primary dependency graph unless
it yields an independently proved delay-scaling result.

## Hard stop conditions

Stop this mainline before a sampled or GPU benchmark if the exact CPU scan
shows any of the following:

- the dynamic safe oracle has negligible value over the best fixed mixture;
- positive value exists only in a hand-selected isolated heterogeneity cell;
- the local shadow is not preserved exactly at zero mixing;
- temporal or cross-agent correlation removes the claimed phase over most of
  the registered grid;
- delay makes the useful region vanish before realistic communication ages.
