# ICML 2027 online joint participation--gain controller

## Executive decision

The recommended mainline is not static q selection, offline joint tuning, or a
generic adaptive optimizer. It is an online data-acquisition and optimization
controller:

> at every decision block, use separately identified dependence and
> learning-state signals to jointly choose how many trajectories to acquire
> and how strongly to update, while a composite Lyapunov drift certifies delay
> stability and resource feasibility.

A working title is:

> **How Many Agents, How Large a Step? Lyapunov Control of Correlated Delayed
> Markov Learning**

The q-only reward-free controller remains a special case and an empirical
foundation. The new main claim is conditional on closing the observability,
predictability, and finite-time proof obligations below.

## Research question

Can a low-complexity predictable controller jointly adapt participation and
gain from observable short Markov trajectories, while achieving a finite-time
near-oracle risk guarantee under cross-agent dependence, delayed updates, and
hard message and actor-transition budgets?

In scope are affine stochastic approximation and fixed-policy TD with a fixed
representation. A learned encoder is an external-validity experiment unless a
separate nonlinear convergence theorem is proved. General Markov games and
actor--critic are not required for the core theorem.

## Why simply adapting q and eta is not enough

Adaptive batch-size methods, adaptive TD gains, and delay-adaptive stochastic
approximation already exist. The novelty cannot be the list of two knobs. It
must be the coupled problem structure:

1. q controls the statistical diversity and resource cost of Markov data, not
   merely an iid minibatch size;
2. cross-agent dependence leaves a nonvanishing common-noise component;
3. eta controls contraction, noise injection, and delay stability;
4. both actions must be predictable from observable data and satisfy two hard
   budgets;
5. the controller must compete with a clairvoyant joint-action policy, not a
   weak fixed-q baseline.

## Controlled delayed recursion

For the lifted TD error, consider

\[
e_{t+1}=e_t-\eta_t A e_{t-D}
+\eta_t\bar\xi_{t+1}^{(q_t)}+r_t,
\]

where `r_t` is the controlled Markov-bias term. Under common/private trajectory
switching,

\[
\operatorname{Cov}(\bar\xi_t^{(q)}\mid\mathcal F_{t-1})
\preceq g(q,\rho)\Sigma_t,
\qquad
g(q,\rho)=\rho+\frac{1-\rho}{q}.
\]

One synchronized round costs `h+q_t` messages and `q_t` actor transitions.
The delay pipeline and every sensing block use the same units and are fully
charged.

## Observable state: the central technical issue

The ideal Lyapunov state

\[
V_t=\|\theta_t-\theta^\star\|_P^2
\]

is unavailable to an executable RL algorithm. Replacing it by the true error
would make the controller an oracle. Replacing it by a raw squared TD residual
would mix Bellman signal and sampling noise and create selection bias.

The proposed solution is orthogonal sensing with disjoint, fully charged short
blocks.

### Structural sensor

State-path fingerprints estimate cross-agent redundancy. Their confidence
sequence gives an interval

\[
I_{\rho,t}=[\underline\rho_t,\overline\rho_t].
\]

This sensor remains reward-free and is independent of the learning outcomes
used to evaluate the selected action.

### Learning-state sensor

Let `F(theta)=A theta-b` denote the TD mean field. Use two conditionally
independent residual blocks at the same predictable parameter iterate:

\[
\widehat F_t^{(1)},\qquad \widehat F_t^{(2)}.
\]

Their cross-product separates signal from zero-mean sampling noise:

\[
\widehat S_t=
\left\langle\widehat F_t^{(1)},\widehat F_t^{(2)}\right\rangle,
\qquad
\mathbb E[\widehat S_t\mid\mathcal F_{t^-}]
=\|F(\theta_t)\|^2
\]

under the registered conditional independence construction. Their difference
provides a scalar noise proxy:

\[
\widehat N_t=
\frac12\left\|\widehat F_t^{(1)}-\widehat F_t^{(2)}\right\|^2.
\]

For strongly monotone affine SA, `||F(theta)||` is equivalent to parameter
error up to the drift condition numbers. Confidence bounds on `S_t` and `N_t`
therefore provide an observable conservative Lyapunov state and noise scale
without estimating a covariance matrix.

This sensor uses TD rewards and is not reward-free. The precise paper claim is
reward-free dependence sensing, not a wholly reward-free joint controller.

## Composite Lyapunov controller

Let the delayed lifted state be

\[
z_t=(e_t,e_{t-1},\ldots,e_{t-D}),
\qquad V_t=z_t^{\mathsf T}Pz_t.
\]

Offline analysis finds `P` and public constants for a safe gain interval
`[eta_min,eta_max(D)]`. Offline certification restricts the action region; it
does not decide the online gain.

Introduce message and environment virtual queues `Q_t^m,Q_t^e`, and define

\[
\Phi_t=V_t+rac{\gamma_m}{2}(Q_t^m)^2
+\frac{\gamma_e}{2}(Q_t^e)^2.
\]

The virtual queues price resource scarcity, while an explicit residual-budget
shield prevents pathwise overspending. A theorem-facing one-step bound should
have the form

\[
\mathbb E[\Delta\Phi_t\mid\mathcal F_t]
\le
-a_D\eta V_t
+\eta^2\{b_DV_t+c_Dg(q,\rho)N_t\}
+Q_t^m(h+q)+Q_t^e q
+\varepsilon_{\rm mix,t}.
\]

Use lower confidence bounds for beneficial drift terms and upper confidence
bounds for noise, correlation, and mixing bias. The resulting robust online
score is observable.

For each candidate q, the quadratic part admits the closed-form safe gain

\[
\eta_t^\star(q)=
\Pi_{[\eta_{\min},\eta_{\max}(D)]}
\frac{a_D\underline V_t}
{2\{b_D\overline V_t+c_D
g(q,\overline\rho_t)\overline N_t\}},
\]

with a predefined fallback when the signal lower bound is zero. The controller
then scans the finite participation catalogue:

\[
q_t\in\arg\min_{q\in\mathcal Q_t^{\rm safe}}
\widehat{\Delta\Phi}_t(q,\eta_t^\star(q)).
\]

Thus q and eta are both selected online. The per-decision arithmetic is
`O(d+|Q|)`, or `O(qd+|Q|)` including the short trajectory summaries. There is
no online LMI, Hessian inverse, covariance matrix, or nested hyperparameter
optimization.

## SDDE interpretation and role

On round time, the controlled delay diffusion is

\[
de(t)=-\eta(t)Ae(t-\tau)dt
+\eta(t)\sqrt{g(q(t),\rho)\Sigma}\,dW(t).
\]

The Lyapunov--Krasovskii functional

\[
\mathcal V(t)=e(t)^{\mathsf T}Pe(t)
+\int_{t-\tau}^{t}e(s)^{\mathsf T}Re(s)\,ds
\]

reveals the same contraction--diffusion--delay tradeoff used by the online
score. It can establish phase geometry, the safe gain envelope, and a
continuous-time oracle. The executed discrete Markov recursion remains the
primary theorem. The SDDE is promoted from interpretation to a formal result
only after proving the interpolation or weak-limit approximation and its error
on the relevant horizon.

## Target theorem stack

### Theorem 1: predictable joint-action stability

For every history-measurable `(q_t,eta_t)` returned by the safe controller,
prove uniform mean-square stability of the delayed lifted recursion and exact
pathwise satisfaction of both budgets.

### Theorem 2: finite-time risk

Bound terminal or Polyak--Ruppert risk with explicit dependence on

\[
\{q_t,\eta_t\}_{t<T},\quad \rho,\quad \tau_{\rm mix},\quad D,
\quad B_m,\quad B_e,
\]

including sensing cost, Markov bias, delay pipeline, and integer horizons.

### Theorem 3: sensing-to-control excess risk

Bound the controller's excess finite risk relative to the same online policy
with access to the true correlation, mean-field norm, and noise scale. The
bound must include both fingerprint and residual-sensor confidence errors.

### Theorem 4: near-clairvoyant joint-control guarantee

On a compact separated class, prove dynamic regret or a constant/logarithmic
matching bound relative to a clairvoyant safe joint `(q_t,eta_t)` policy under
the same budgets. A theorem only relative to the best q with a fixed eta is not
sufficient for the upgraded claim.

### Theorem 5: strict value of joint control

Construct a delayed correlated Markov family in which every q-only and every
eta-only controller has a strictly worse finite-budget risk than joint control.
This theorem answers the reviewer question of why two online actions are
scientifically necessary rather than a cosmetic combination.

## Experimental standard

The primary baselines must receive identical sensing and resource accounting:

1. best task-by-budget fixed `(q,eta)` selected on disjoint seeds;
2. q-only online controller with a strong fixed gain;
3. eta-only adaptive TD with a strong fixed q;
4. adaptive batch/gain method that ignores cross-agent dependence;
5. true-state joint oracle under identical budgets;
6. the proposed observable joint controller.

Main metrics are terminal prediction risk, normalized learning AUC, CVaR90,
strict improved-cell breadth, budget utilization, and controller overhead.
The new controller must beat the best fixed pair and both one-dimensional
adaptive ablations, not merely all-agent participation.

The current MinAtar fixed-representation tasks are suitable for the first
prospective CPU study. A later GPU experiment may train the representation or
use a standard deep value learner as external validity, but it cannot replace
the theorem-aligned experiment.

## Outcome-free gates before a new scientific pilot

1. The observable residual cross-product identities pass analytic and Monte
   Carlo unit tests under Markov blocking.
2. A common or switching Lyapunov certificate covers the entire online gain
   interval for every registered delay.
3. The robust score is predictable and cannot use the outcome produced by the
   action it selects.
4. The residual-budget shield proves pathwise message and environment
   feasibility, including probe and delay costs.
5. A frozen fixed-action scan shows a meaningful oracle ceiling over the best
   fixed `(q,eta)`, with predeclared aggregate and breadth thresholds.
6. The online complexity and memory bounds are verified by tests.
7. Only after these gates pass are new pilot seeds, a runner, and an analyzer
   preregistered.

No GPU is needed for these theory and CPU feasibility gates.

## Devil's-advocate audit

The strongest attack is that the controller may secretly use an unobservable
error or a biased residual proxy. That would invalidate the claimed online
algorithm even if experiments were positive. The paired, disjoint residual
sensor and its confidence theorem are therefore claim-critical, not an
optional implementation detail.

The second attack is novelty by combination: adaptive batch size and adaptive
gain are individually established. The defense must be a theorem showing that
cross-agent redundancy, delay, and dual resource prices create a joint-control
phase and a strict separation from q-only and eta-only policies.

The third attack is scope inflation. Fixed-policy TD with parallel samplers is
not a general cooperative Markov game. The paper should use multi-agent Markov
learning or parallel-agent RL language unless a genuine actor--critic or Markov
game theorem and experiment are added.

## Decision checkpoint

This direction is stronger than both q-only selection and offline schedule
selection, but it is not yet authorized for scientific execution. The next
work item is a CPU theory prototype for the observable residual identities,
the robust one-step drift minimizer, and the common delayed Lyapunov
certificate. Failure of observability or joint-action value should stop the
upgrade and preserve the current q-only result rather than weaken the gates.
