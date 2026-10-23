# Learning Without Spoiling Rollouts: Lyapunov Policy-Inventory Control for Asynchronous MARL

Status: candidate theory interface and outcome-free algebra only.  No efficacy
experiment, formal seed, standard benchmark or GPU job is authorized.

## One problem, one story

In centralized training with decentralized execution, heterogeneous actor
workers collect joint trajectories under snapshots of a factorized joint
policy.  While one paid trajectory is still in flight, updates to any teammate
change the target joint policy.  The trajectory does not merely become “old”:
its importance weights become more variable, so a step that helps immediately
can spoil many future gradient estimates.

The central question is therefore:

> How should an asynchronous MARL trainer make each policy update when that
> update produces current learning progress but devalues every paid rollout
> that is still in flight?

This is a stochastic policy-inventory problem.  The inventory consists of
rollouts already launched but not yet consumed.  The online action is the
admission and continuous size of the current block-policy update.  The state is
not an abstract delay counter: it is the measurable off-policy second-moment
risk of every outstanding rollout.  Strict CTDE is retained; execution uses the
same decentralized policies and adds no communication.

## Exact multi-agent inventory geometry

Let the current joint policy and the behavior policy of outstanding rollout
`p` factorize as

\[
\pi_\theta(a\mid o)=\prod_{j=1}^n\pi_{j,\theta_j}(a_j\mid o_j),
\qquad
\mu_p(a\mid o)=\prod_{j=1}^n\mu_{p,j}(a_j\mid o_j).
\]

For fixed-variance Gaussian blocks in the one-step subclass, write
`mu_{p,j}=N(theta^b_{p,j},sigma_j^2 I)` and
`pi_j=N(theta_j,sigma_j^2 I)`.  The joint likelihood ratio `W_p` obeys the
exact identity

\[
z_p(\theta)
:=\log\mathbb E_{\mu_p}[W_p^2]
=\sum_{j=1}^n
\frac{\|\theta_j-\theta^b_{p,j}\|^2}{\sigma_j^2}.
\tag{1}
\]

Thus `exp(z_p)-1` is exactly the chi-square variance inflation and the log risk
is additive across agents.  For finite-horizon Markov games, the same object is
defined on trajectory likelihood ratios; a theorem must use either an exact
conditional Renyi chain rule or an explicitly stated uniform conditional
upper bound.  The Gaussian identity is not silently promoted to arbitrary
neural policies.

If the arriving update changes block `i` by `alpha g_i`, every remaining
inventory item changes exactly as

\[
z_p(\alpha)=z_p(0)
+\frac{2\alpha\langle\theta_i-\theta^b_{p,i},g_i\rangle}{\sigma_i^2}
+\frac{\alpha^2\|g_i\|^2}{\sigma_i^2}.
\tag{2}
\]

Equation (2) captures the causal externality missing from scalar age rules:
the same step can spoil one behavior batch while moving closer to another.

## The physical Lyapunov function

Let `J(theta)` be the common cooperative return.  For the in-flight set
`F_k`, define

\[
\mathcal L_k
=V\bigl(J^\star-J(\theta_k)\bigr)
+\sum_{p\in\mathcal F_k}w_p\bigl(e^{z_{p,k}}-1\bigr).
\tag{3}
\]

The second term is not a virtual proxy.  In the Gaussian subclass it is the
exact importance-weight variance inflation of paid data.  More generally it is
an upper bound that must be established from declared policy-ratio conditions.

Suppose completed rollout `i` supplies an importance-corrected block-gradient
estimate `gtilde_i`.  If each
trajectory-gradient contribution has norm at most `C_i` and the batch has
`B_i` independent episodes, the second-moment interface gives

\[
r_i(z_i)\le \frac{C_i e^{z_i/2}}{\sqrt{B_i}}.
\tag{4}
\]

This is only an RMS statement and cannot be inserted as a realized norm-error
radius after observing the same batch.  The executable gain certificate below
requires a simultaneous norm-error event
`||gtilde_i-nabla_i J||<=r_i(z_i)`, obtained from bounded ratios with explicit
clipping bias or from a valid robust mean theorem.  Conditional on that event,
block smoothness yields

\[
G_i(\alpha)
=\alpha\bigl(s_i^2-r_i(z_i)s_i\bigr)
-\frac{L_i}{2}\alpha^2s_i^2,
\qquad s_i=\|\widetilde g_i\|.
\tag{5}
\]

Consuming rollout `i` removes its inventory risk.  Combining (2)--(5) gives
the executable one-event drift envelope

\[
\overline\Delta_i(\alpha)
=-V G_i(\alpha)-w_i(e^{z_i}-1)
+\sum_{p\ne i}w_p
\left[e^{z_p(\alpha)}-e^{z_p(0)}\right].
\tag{6}
\]

Because each `z_p(alpha)` is a convex quadratic and the exponential is
nondecreasing and convex, (6) is a scalar convex function.  Its derivative is
monotone.  The update and its size are selected jointly by

\[
\alpha_k\in\arg\min_{0\le\alpha\le\bar\alpha_i}
\overline\Delta_i(\alpha),
\tag{7}
\]

using endpoint checks and bisection.  Complexity is `O(P)` inventory terms plus
`O(log(1/epsilon))` scalar iterations for at most `P` outstanding rollouts.
There is no Hessian, preconditioner, covariance inverse, finite learning-rate
catalogue or participation scan.

## Required theorem chain

The route is viable only if one analysis closes all of the following for the
same executable algorithm.

1. A factorized Markov-trajectory second-moment or clipped-ratio bound that
   makes `z_p` observable and covers the actual gradient estimator.
2. A telescoping drift theorem controlling both optimization error and the
   variance inflation of completed rollouts.
3. A finite-wall-clock stationarity bound under heterogeneous Markov service,
   with a nonzero-update condition rather than a shield that always returns
   zero.
4. A separation family in which every fixed step/staleness threshold either
   wastes low-load progress or spoils high-load inventory, while (7) adapts
   causally.
5. A low-load corollary recovering the base optimizer and an explicit
   unavoidable tradeoff where universal no-harm is impossible.

The SDDE view is optional.  A marked-point-process or SDDE limit may explain
how arrival intensity, agent count and policy drift create a variance phase
transition, but the discrete event-time theorem remains the guarantee.

## Novelty boundary and experimental ladder

V-trace/MA-Trace-style importance correction, generic asynchronous SGD,
gradient alignment, ESS-aware learning-rate scaling and delay compensation are
inherited components, not claimed contributions.  The candidate contribution
is the causal pricing of how a current distinct-agent policy update changes the
factorized off-policy risk of **all paid outstanding joint rollouts**, together
with a wall-clock MARL guarantee and a low-complexity executable controller.

The evidence ladder is deliberately short:

1. exact identities, convexity, estimator-risk and theorem audit;
2. outcome-free CPU oracle headroom on a factorized Gaussian cooperative game
   with bursty heterogeneous completion processes;
3. only if both pass, disjoint stochastic CPU pilot and formal confirmation;
4. only after CPU confirmation, asynchronous MAPPO/MA-Trace-style standard
   cooperative MARL benchmarks with fully charged wall-clock and rollout cost.

The stopped T-083A, PUB and unconditional-transport outcomes remain unchanged
and are not confirmation evidence for this candidate.
