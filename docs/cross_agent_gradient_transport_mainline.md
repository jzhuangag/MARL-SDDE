# Transport, Don't Discard: Lyapunov-Certified Cross-Agent Gradient Transport for Asynchronous MARL

Status: outcome-free mainline design and algebra gate; no efficacy experiment is authorized by this document.

## One research question

Can a centralized trainer retain the wall-clock advantage of asynchronous heterogeneous multi-agent rollouts without applying policy gradients that have expired because teammates updated while those rollouts were in flight?

This is the sole problem statement.
It is not participation control, graph selection, online learning-rate search, safe constrained MARL, or asynchronous action execution.
The agents have distinct policies in one cooperative Markov game.
Training is centralized and asynchronous; execution remains decentralized and requires no new communication.

## Why the problem is intrinsically multi-agent

Let the joint policy be `theta=(theta_1,...,theta_n)` and let `J(theta)` denote the common discounted return.
Agent `i` starts an on-policy block-gradient computation at joint policy `theta^b` and returns later, after other policy blocks have changed.
Its raw proposal estimates

\[
g_i^b=\nabla_i J(\theta^b),
\]

but the server needs `nabla_i J(theta^k)`.
In a distinct-policy Markov game, the discrepancy contains cross-agent blocks

\[
\nabla^2_{ij}J(\theta^b)(\theta_j^k-\theta_j^b),\qquad j\ne i.
\]

These terms are absent from a model in which agents are merely independent workers estimating one shared policy gradient.
They express a training externality: improving teammate `j` changes whether agent `i`'s already-paid rollout still points uphill.

The standard compromises all waste something important.
A barrier wastes wall-clock time on stragglers; raw asynchronous application wastes stability; age clipping or discarding wastes trajectories; shrinking every delayed step wastes progress even when the cross-agent change is predictable.

## Core operation: cross-agent gradient transport

When a proposal arrives, compute one joint Hessian-vector product on its retained rollout/surrogate and transport it to the current joint policy:

\[
\widetilde g_i^k
=\widehat g_i^b
+\widehat H_{i,:}^b(\theta^k-\theta^b).
\tag{1}
\]

This is one continuous correction, not a scan over agent counts or learning-rate catalogues.
Automatic differentiation evaluates the Hessian-vector product without forming or inverting a Hessian matrix.
The intended neural implementation charges the extra backward pass and retains decentralized execution.

Assume on a declared trust region that the block Hessian is `rho_i`-Lipschitz, the gradient estimator has radius `r_i^g`, and the Hessian-vector product has operator radius `r_i^H`.
Taylor's theorem gives the executable radius

\[
\left\|\widetilde g_i^k-\nabla_iJ(\theta^k)\right\|
\le
R_i(\Delta_i)
:=r_i^g+r_i^H\|\Delta_i\|
+\frac{\rho_i}{2}\|\Delta_i\|^2,
\quad
\Delta_i=\theta^k-\theta^b.
\tag{2}
\]

Without transport, smoothness gives a first-order stale radius `r_i^g+L_i||Delta_i||`.
With exact Hessian-vector products, transport changes the delay term from first to second order.
For a quadratic potential game, `rho_i=0` and (1) is exact regardless of cross-agent coupling or delay.
This directly targets the failure mode of the stopped PUB controller, whose magnitude-only first-order debt remained large even when the cross change was exactly predictable.

## Certified progress and the Lyapunov function

Let `s_i=||tilde g_i^k||`.
Block smoothness gives

\[
J(\theta^k+\alpha U_i\widetilde g_i^k)-J(\theta^k)
\ge
\alpha(s_i^2-R_i s_i)
-\frac{L_i}{2}\alpha^2s_i^2
=:G_i(\alpha).
\tag{3}
\]

If this were the only term, the exact continuous optimum would be

\[
\alpha_i^{\rm gain}
=\left[\frac{1-R_i/s_i}{L_i}\right]_{[0,\bar\alpha_i]}.
\tag{4}
\]

The update also changes the transport radius of proposals still in flight.
For each such proposal `p`, let `ell_p` be its current joint-policy path length and let `Z_p=R_p(ell_p)` be its certified residual radius.
For a candidate displacement of length `d=alpha s_i`, the outcome-free radius increment obeys

\[
a_{p,i}(\alpha)
\le
(r_p^H+\rho_p\ell_p)d+\frac{\rho_p}{2}d^2.
\tag{5}
\]

Use the composite Lyapunov function

\[
\mathcal L_k
=V\bigl(J^\star-J(\theta^k)\bigr)
+\frac12\sum_{p\in\mathcal F_k}w_p Z_{p,k}^2,
\tag{6}
\]

where `F_k` is the set of in-flight proposals.
The first term is optimization error and the second is the future value already at risk in paid rollouts.
It is not an arbitrary virtual queue: every `Z_p` upper-bounds a concrete pending-gradient error.

Combining (3)-(6) yields the one-event drift envelope

\[
\overline\Delta_i(\alpha)
=-V G_i(\alpha)-\frac{w_i}{2}Z_i^2
+\frac12\sum_{p\ne i}w_p
\left[(Z_p+a_{p,i}(\alpha))^2-Z_p^2\right].
\tag{7}
\]

Every coefficient in `a_{p,i}` is nonnegative.
Consequently (7) is a scalar convex polynomial of degree at most four on the public trust interval.
Its derivative is monotone, so admission and step size are obtained jointly by one bisection or safeguarded Newton solve.
There is no finite hyperparameter scan, covariance inverse, Hessian matrix, or quadratic program.
Per completion, bookkeeping is `O(n)` plus one Hessian-vector product and `O(log(1/epsilon))` scalar iterations.

When the objective is quadratic and the Hessian-vector product is exact, every future-radius increment in (5) is zero.
Then (7) reduces to the gain maximizer (4), rather than the overly conservative first-order PUB price.

## Target main theorem

The paper is viable only if one theorem proves the following chain for the same executable algorithm.

1. **Transport coverage.** Under explicitly stated episodic Markov-gradient and Hessian-vector confidence events, (2) holds uniformly over all accepted completions.
2. **Lyapunov descent.** The continuous minimizer of (7) gives a telescoping bound on optimization error plus outstanding transported-proposal error.
3. **Event-time stationarity.** For stochastic block policy gradients, the average full joint-gradient norm reaches a floor determined by gradient/HVP estimation and quadratic Taylor remainder, with the delay dependence entering through transported path energy rather than raw maximum age alone.
4. **Wall-clock conversion.** Under a separately stated Markov completion condition, the event-time rate converts to elapsed training time and exposes the straggler benefit.
5. **Separation.** A theorem-defined heterogeneous-delay family gives a strict wall-clock gap over barrier, raw stale, age-discard and first-order delay scaling, while a low-delay corollary recovers the base optimizer.

The claim must be a stationarity or potential-improvement result under the declared cooperative Markov-game assumptions.
It must not silently become global neural-policy optimality.

## What is and is not proved now

The deterministic algebra audit currently proves only the following interfaces:

- one-HVP transport is exact on every tested coupled quadratic system;
- the nonlinear Taylor remainder is covered by the `rho_i ||Delta||^2/2` radius on the declared quartic family;
- the transported radius is strictly smaller than the raw first-order radius throughout the local audit grid;
- the continuous Lyapunov step matches a dense numerical optimizer.

These are formula checks, not efficacy evidence.
The Markov policy-gradient/HVP confidence theorem, complete telescoping argument, strong-comparator separation and CPU oracle-headroom gate remain mandatory before any new pilot.

## Experimental ladder and stopping rules

The experimental story, if the theorem closes, has only three layers.

1. **Outcome-free algebra and exact games.** Verify transport coverage, exact quadratic cancellation, scalar optimization and complexity.
2. **Disjoint CPU mechanism confirmation.** Use stochastic nonlinear potential games with Markov rollout latency, fully charge HVP and discarded-trajectory cost, and compare against fresh serial, barrier, raw async, age/ratio discard, DC-ASGD-style compensation and delay-adaptive step baselines.
3. **Standard cooperative MARL.** Use centralized training with decentralized execution, heterogeneous rollout workers and distinct actors; report sample efficiency, wall-clock return, gradient stationarity proxies, discarded trajectories, HVP overhead and scaling in agent count/delay.

No CPU efficacy population is authorized until the full performance bound and an outcome-free oracle-headroom calculation both pass.
No GPU benchmark is authorized until a disjoint CPU pilot and formal confirmation pass their frozen gates.

## Scope decision

This mainline is materially deeper than selecting participation or learning rate.
It identifies a multi-agent optimization object that existing delay counters miss: the block-Hessian transport of one agent's gradient through teammates' policy updates.
Lyapunov drift prices the residual value of all paid in-flight rollouts and produces one continuous low-complexity update.
SDDE is optional as a continuous-time interpretation of the transported delayed dynamics; it is not required for the core theorem and will not be used as decorative language.
