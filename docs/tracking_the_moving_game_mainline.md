# Tracking the Moving Game: a coupled actor--critic successor audit

Date: 2026-09-05.

Status: **theory-interface candidate only**.  This note does not reopen the
failed Two Clocks public-MPE bridge, authorize a new efficacy experiment, or
turn any old development population into confirmation evidence.

## Decision

The frozen Two Clocks results rule out a paper whose only practical action is
a conservative scalar actor step.  The finite-game rate--coupling theorem and
its positive exact/stochastic evidence remain correct within their stated
scope, but the public-MPE bridge produced only `0.492%` heterogeneous AUC gain
and failed taskwise learning/phase gates.  Replacing its Monte-Carlo baseline
by HAA2C without changing the research question would be an unregistered
performance rescue, not a new contribution.

The only successor worth a bounded CPU feasibility test is:

> **In asynchronous CTDE, how should a learner couple policy progress to the
> work needed for a centralized critic to track a target that the same
> asynchronous policy updates keep moving?**

This is a task-level change.  Asynchrony creates two causally coupled debts:

1. an arriving owner packet is self-fresh but stale with respect to teammate
   policy motion; and
2. applying that packet moves the value target seen by the centralized
   critic, while critic error biases later policy updates.

A fixed actor/critic timescale ratio must compromise across these phases.  A
candidate **Lyapunov-coupled actor--critic** rule instead chooses the actor
application scale and critic correction scale together at every arrival.  The
primary claim would be finite-resource learning risk versus charged Markov
transitions and learner updates.  Wall-clock time remains an important
secondary axis, not the sole definition of success.

Training is centralized and event driven.  Each agent owns a distinct actor
block; a centralized critic may use joint information during training.
Execution remains decentralized and requires no new communication.

## Exact minimal model

The following model is deliberately small enough that the design action and
the Lyapunov drift can be checked exactly before any stochastic trajectory is
sampled.  Let a quadratic potential gap be

\[
 f(x)=\tfrac12 x^\top Hx,\qquad H=H^\top\succ0,
\]

where `x_i` is the error of actor block `i`.  Let `e` be the centralized
critic's tracking error and use

\[
 \mathcal V(x,e)=x^\top Hx+\omega e^2.                 \tag{1}
\]

When owner `i` returns, its stored packet was born at teammate snapshot
`\bar x`.  Single-flight ownership keeps `x_i=\bar x_i`; only off-diagonal
coordinates may be stale.  The packet direction is

\[
 g_i^{\rm pkt}=H_{ii}x_i+\sum_{j\ne i}H_{ij}\bar x_j+B_i e.    \tag{2}
\]

The current game gradient is `(Hx)_i`.  Their difference therefore separates
critic error from strategic staleness exactly:

\[
 g_i^{\rm pkt}-(Hx)_i
 =B_i e+\sum_{j\ne i}H_{ij}(\bar x_j-x_j).              \tag{3}
\]

At the event, choose actor scale `alpha` and critic scale `beta`:

\[
 \begin{aligned}
 x_i^+&=x_i-\alpha g_i^{\rm pkt}-\alpha\xi_a,\\
 e^+&=e-\nu\beta e+\kappa_i\alpha g_i^{\rm pkt}
       +\kappa_i\alpha\xi_a+\beta\xi_c .               \tag{4}
 \end{aligned}
\]

The `kappa_i` term is not decorative: changing actor `i` moves the critic's
fixed point, so even a perfect pre-update critic acquires tracking error.  The
zero-mean innovations have variances `sigma_a^2` and `sigma_c^2` and are
conditionally independent.

For `y=(x,e)`, define the two response vectors

\[
 a=(-g_i^{\rm pkt}{\bf e}_i,\;\kappa_i g_i^{\rm pkt}),\qquad
 b=(0,\ldots,0,-\nu e),
\]

and `P=diag(H,omega)`.  Expanding (1)--(4) gives the exact conditional drift

\[
 \mathbb E[\mathcal V^+-\mathcal V\mid\mathcal F]
 =h^\top u+\tfrac12u^\top Q u,\qquad u=(\alpha,\beta),   \tag{5}
\]

with

\[
 h=2\begin{bmatrix}a^\top Py\\b^\top Py\end{bmatrix},
\quad
 Q=2\left(
 \begin{bmatrix}a^\top Pa&a^\top Pb\\a^\top Pb&b^\top Pb\end{bmatrix}
 +\begin{bmatrix}
 \sigma_a^2({\bf r}_a^\top P{\bf r}_a)&0\\
 0&\sigma_c^2\omega
 \end{bmatrix}\right),                                 \tag{6}
\]

where `r_a=(-e_i,kappa_i)`.  The matrix `Q` is positive semidefinite because
it is a Gram matrix plus nonnegative noise curvature.  Consequently

\[
 (\alpha_k,\beta_k)
 =\arg\min_{0\le\alpha\le\bar\alpha,
            0\le\beta\le\bar\beta}
 h_k^\top u+\tfrac12u^\top Q_k u                       \tag{7}
\]

is a convex two-variable box QP.  Its global solution is obtained by checking
the unconstrained stationary point and four one-dimensional faces.  This is
constant-size control overhead; constructing the policy and critic gradients
still dominates the cost.

Equation (7) is the required role of Lyapunov as a **design tool**.  It is not
an offline catalogue, a finite scan over agent counts, or a queue added after
an algorithm has already been chosen.  The off-diagonal entry of `Q` couples
the two actions through critic-target motion.  Removing it produces a strong
online but incorrectly decoupled comparator, not merely a fixed learning-rate
baseline.

## What the minimal derivation proves and does not prove

Equations (2)--(7) prove one exact algebraic fact on the declared quadratic
model: the joint action is a low-complexity convex Lyapunov minimization and
the teammate-staleness and critic-target terms are identifiable separately.
They do **not** yet prove a MARL algorithm.

The executable nonlinear interface still needs observable, non-outcome-
selected bounds for:

- the current policy-gradient signal in place of the unknown actor error;
- critic tracking error from mandatory TD residuals without treating a
  function-approximation residual as value error for free;
- off-diagonal policy motion from the recorded version path;
- Markov mixing, truncation and importance-ratio bias of a stored packet; and
- critic fixed-point sensitivity `kappa_i`.

A valid general theorem must telescope the same implemented `(alpha,beta)`
actions and convert its drift to a potential/Nash or standard stationarity
criterion.  The first executable route pays recorded policy and critic
version displacement once through robust packet radii and therefore sets the
optional Lyapunov--Krasovskii history term to zero.  A later history-energy
refinement may replace that robust term but cannot duplicate it.  No neural
claim may inherit (5) just by calling TD loss `e^2`.

## Novelty confrontation

The candidate is narrower than, and must be explicitly separated from, five
nearby lines.

- Chen and Zhao's [ICML 2025 single-timescale actor--critic
  analysis](https://proceedings.mlr.press/v267/chen25co.html) already uses one
  Lyapunov framework to analyze coupled actor and critic iterates under
  Markovian sampling.  The possible new object here is not a composite
  Lyapunov function by itself; it is the online two-action minimizer under
  distinct delayed policy blocks and a moving centralized-critic target.
- D'Andrea and Light's August-2026
  [finite-time asynchronous MARL actor--critic](https://arxiv.org/abs/2608.22840)
  makes a generic claim of finite-time asynchronous two-timescale learning
  non-novel.  The present route survives only if training-completion delay,
  owner self-freshness, teammate strategic staleness and the controlled critic
  target appear in a sharper theorem and in the executable rule.
- Mahadevan et al.'s [two-timescale stochastic approximation under Markovian
  noise](https://arxiv.org/abs/2605.31172) already addresses stability and
  convergence of general two-timescale Markov SA.  A proof that is just a
  corollary of that result must be stopped.
- Xiao, Tan and Amato's [asynchronous MARL
  actor--critic](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1c153788756d35559c22d105d1182c30-Abstract-Conference.html)
  concerns asynchronous action durations.  This candidate concerns
  asynchronous training packet completion under otherwise ordinary
  decentralized execution.
- Shared-model delayed SGD and its SDDE scheduling theory already optimize
  worker/group/dropout policies.  Distinct strategic actor ownership and the
  critic target feedback must remain essential after the proof is simplified.

The novelty risk is therefore **high but testable**.  “Lyapunov actor--critic”
or “adaptive actor/critic learning rates” alone is not an ICML-level claim.

## Two pre-experiment kill gates

No new sampled trajectory or GPU job is authorized until both gates pass.

1. **Performance-bound gate.**  Derive a conditional finite-time inequality
   for a declared tabular/linear-critic Markov-potential-game class in which
   the implemented joint action (7), the in-flight history term and critic
   tracking term telescope together.  It must not assume access to `x` or `e`
   in the executable algorithm.
2. **Oracle-headroom gate.**  On a frozen, outcome-free grid of exact
   multi-block systems, the joint two-action oracle must materially outperform
   both the per-scenario best fixed `(alpha,beta)` pair and the online
   diagonalized Lyapunov rule.  Improvement only over a bad fixed pair is not
   sufficient.  Separable/no-staleness controls must expose any unnecessary
   cost.

Passing both gates would authorize a separately frozen CPU stochastic
confirmation.  Only a successful independent confirmation would justify a
standard HAPPO/MAPPO/HAA2C GPU preregistration.  Failure of either gate stops
this successor rather than generating another controller variant.

### Oracle-headroom outcome

The separately preregistered exact-moment scan at commit `fc8646b` passed
H1--H10 and reproduced byte for byte.  The coupled/best-fixed geometric AUC
ratio is `0.732823`, and the coupled/online-diagonal ratio is `0.948118`, with
all 128 primary scenarios directionally favorable.  Target-motion-zero
controls reduce exactly to the diagonal rule.  This closes only kill gate 2.

Kill gate 1 remains open.  The observable two-action drift, finite-schedule
reset-trajectory concentration, stale linear-critic contraction, and
single-counted version-path interface are now proved under explicit
conditions.  The remaining blockers are instantiating nonvacuous constants,
activated-owner/game conversion, and a comparison theorem for the same
executable algorithm.  Sampled CPU efficacy, formal seeds and GPU work remain
prohibited.  Full oracle results and limitations are in
`validation_coupled_actor_critic_headroom.md`.

## Experimental package if the gates survive

The eventual paper must report return/potential gap against four resources:
charged environment transitions, actor optimizer steps, critic optimizer
steps/FLOPs, and wall-clock time.  Required strong comparators are a tuned
fixed actor/critic ratio, the same asynchronous learner with the cross action
deleted, a generic delay-aware asynchronous rule, a fully utilized barrier,
and the native synchronous CTDE algorithm.  Main-text evidence needs multiple
standard task families and both favorable and unfavorable service--coupling
regimes.

SDDE is optional.  It may summarize the small-step random-delay limit and
predict a phase boundary, but the discrete event-time Lyapunov theorem and
the fully charged experiments are primary.
