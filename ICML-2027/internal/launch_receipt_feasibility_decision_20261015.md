# Bounded successor-design feasibility decision

## Material Passport

Date: 2026-10-15. Input commit: `5112e24dfddbeccfc6afa6a97745161bd9f82fa2`.
Role: theoretical and analytic design qualification after the complete A1
study; no new scientific trajectory, performance experiment, or manuscript edit.
Evidence: an explicit launch/receipt derivation, an executable promise objective,
a distribution-free precision preflight, and fresh primary-source checks.

## Decision

**No-go for a successor RL experiment using the current bounded-gradient
certificate.** Neither the fixed-launch receipt-step design nor the two-stage
launch/receipt design currently meets the nonvacuity and novelty entry gates.
The resource timing gap can be closed conditionally by an in-flight progress
potential, derived below. Its observable confidence radius is too large at the
16-episode budget: it forces every certified action to have zero step even in
the most favorable zero-staleness case.

This is a decision about this particular sufficient certificate. It is not an
information-theoretic impossibility statement about sharper estimators or
asynchronous MARL. Increasing the batch size from the analytic table is not an
approved experiment; the table gives necessary, not sufficient, conditions.

| Design | Executable variable | Correct cost timing | Qualified guarantee | Entry decision |
|---|---|---|---|---|
| Fixed launches, controlled receipt steps | Step for each eligible received raw gradient | Launch work and transmitted packets remain charged after rejection | Conditional return-increment lower bound | No-go: current bound returns zero steps; a generic delay-adaptive step is not a sufficient novelty claim |
| Controlled launches and precommitted steps | Launch subset and a step fixed before the job's gradient is revealed | Each launch reserves compute/environment/communication, completion transmits once | Conditional finite-horizon promised-return accounting and bounded resource queues | No-go: launch/receipt mathematics closes under explicit assumptions, but the current confidence radius leaves no positive promise |

## 1. Exact stochastic-gradient interface

Let `J(theta)` be the expected native finite-horizon team return for distinct
interacting policies, with `J_min <= J <= J_max`. Write `B_J=J_max-J_min`.
A job `j` for actor `i` launches at `s_j` with frozen joint policy `theta_s`.
It produces an average `g_j` of `m` independent **full joint episodes** satisfying

\[
\mathbb E[g_j\mid\mathcal F_{s_j}]=\nabla_iJ(\theta_{s_j}),
\qquad \|g_j\|\le G_i.
\]

Each per-episode gradient is also bounded by `G_i`. Jobs use disjoint episode
randomness, their samples and gradients remain hidden until completion, and
completion/order do not depend on the unrevealed gradient. This is an exact
likelihood-ratio team-return gradient assumption. A PPO clipped-objective
gradient, critic-loss gradient, GAE approximation, gradient clipping, or an Adam
parameter increment does not automatically satisfy it. A biased estimator
requires an additional proved bias allowance.

Independence is between episodes and jobs, not between timesteps within an
episode. The within-episode Markov law remains part of `J`. Sharing one rollout
batch across jobs would require a separate conditional-dependence argument:
revealing a correlated actor gradient can change the conditional mean of an
unrevealed packet. Merely assigning different packet IDs does not resolve it.

Native sampling is charged for all agents in every joint episode. Separate
per-job episodes are separate paid trajectories. The successor event model
uses single-use job IDs, server-owned optimizer state, and an exogenous service
sequence indexed by actor and launch ordinal. The derivation below uses raw
gradient ascent `theta_i <- theta_i + alpha_j g_j`, not stale Adam-state replacement.

## 2. Fixed-launch receipt-step design

Suppose a simultaneous source-gradient confidence event gives
`||g_j - grad_i J(theta_s)|| <= r_j`. A uniform block-gradient Lipschitz
constant `L_i` then gives, at receipt policy `theta_t`,

\[
d_j=r_j+L_i\|\theta_t-\theta_s\|,
\qquad
\langle\nabla_iJ(\theta_t),g_j\rangle
\ge \|g_j\|^2-d_j\|g_j\|.
\]

For an `L`-smooth team-return objective, the actual update satisfies

\[
J(\theta_t+\alpha g_j)-J(\theta_t)
\ge \alpha A_j-\tfrac L2\alpha^2\|g_j\|^2,
\quad A_j=\|g_j\|^2-d_j\|g_j\|.
\]

The maximizing step is `clip(A_j/(L ||g_j||^2),0,alpha_max)` when the denominator
is positive; zero gradient gives zero step. This is an executable design rule
when the source confidence and smoothness constants are valid. The probability
event must cover every inspected packet; selecting a step after looking at the
packet does not authorize a marginal, post-selected interval.

Acceptance cannot refund already paid work or communication. Consequently,
receipt control does not solve the launch-resource allocation problem.

## 3. An in-flight potential closes the delayed launch timing

At launch, use a **previously completed** estimate `z_j` from policy `theta_0`,
with a source-gradient confidence radius `r_j`. On its good event, define

\[
e_j=r_j+L_i\|\theta_s-\theta_0\|,
\quad a_j=(\|z_j\|-e_j)_+,
\quad b_j=\min\{G_i,\|z_j\|+e_j\}.
\]

The new source gradient `q_j=grad_i J(theta_s)` then obeys
`a_j <= ||q_j|| <= b_j`. Fix a valid receipt-motion bound `R_j` covering all
other policy updates before this job is applied. One explicit conservative
choice is `R_j=n D alpha_max G_max` if each actor has at most one in-flight job,
service takes at least one and at most `D` ticks, all due arrivals are processed
in a public order before any new launch at that tick, and every applied increment has norm at most
`alpha_max G_max`. Holding ready jobs beyond that deadline invalidates this choice.

Select `alpha_j` at launch, before observing this job's new gradient. Disjoint
hidden job samples preserve its conditional mean until its public receipt.
Using smoothness and the bound on intervening policy motion yields

\[
\mathbb E[\Delta J_j\mid\text{receipt history before revealing }g_j]
\ge h_j(\alpha_j)
=\alpha_j(a_j^2-L_iR_jb_j)
-\tfrac L2\alpha_j^2G_i^2
\]

on the old estimate's good event. The lower bound follows from
`<grad_i J(theta_t),q_j> >= ||q_j||^2-L_i R_j ||q_j||` and
`E||g_j||^2 <= G_i^2`. Its maximizing bounded step is closed form. Allowing
zero makes the chosen promise `h_j >= 0`.

This expectation argument does **not** permit examining `g_j` and then deciding
whether to apply the precommitted step. Such selective acceptance changes the
conditional expectation and needs the different uniform, pathwise certificate
from Section 2. The two proofs cannot be interchanged.

Let `P_t` be the sum of these fixed nonnegative promises for jobs still in
flight. Introduce launch resource queues

\[
Q_r(t+1)=[Q_r(t)+c_r(x_t)-\bar c_r]^+,
\qquad
\Phi_t=-vJ(\theta_t)-vP_t+\tfrac12\sum_rQ_r(t)^2.
\]

A new launch adds `h_j` to `P_t`, producing drift `-v h_j`. At a completion,
removing the promise contributes `+v h_j`; the expected actual learning
improvement contributes at most `-v h_j`. These terms cancel. This is the
missing delayed-work accounting: launch receives a debit against a certified
future improvement, rather than pretending the improvement occurs immediately.

With bounded one-tick resource costs, the launch-dependent drift bound is

\[
\mathbb E[\Delta\Phi_t]
\le C+
\mathbb E\!\left[\sum_r Q_r(t)(c_r(x_t)-\bar c_r)
-v\sum_{j\text{ launched at }t}h_j\right]
+\text{confidence-failure allowance}.
\]

Thus the design objective is to maximize
`v sum_launch h_j - sum_r Q_r c_r(x)` over currently available actors, resource
caps, and their precommitted steps. `launch_promise_feasibility.py` implements
the closed-form steps and exact small-team launch-subset objective.

### Finite-horizon interpretation and confidence failures

Suppose at most `M` jobs launch and the source certificates hold simultaneously
with probability at least `1-delta`, including all adaptive inspections and
reuse. Let `0 <= h_j <= h_max`. On a bad source certificate the return increment
remains at least `-B_J`. The bad indicator for each job is measurable before
its new gradient is generated. A global confidence failure can affect every
job, so the conservative expected number of affected jobs is at most `M delta`.
Without conditioning future gradient noise on an entire future confidence event,

\[
\mathbb E[J(\theta_T)-J(\theta_0)]
\ge\mathbb E\sum_{j\text{ completed by }T}h_j
-(B_J+h_{\max})M\delta
\ge\mathbb E\sum_{j\text{ launched by }T}h_j
-n h_{\max}-(B_J+h_{\max})M\delta.
\]

The terminal `n h_max` term follows from at most one outstanding job per
actor and immediate receipt processing. Reusing an estimate does not create
independent certificates. A sharper `delta`-only allowance requires a valid
predictable per-job allocation or a one-use certificate argument; it does not
follow from the global confidence event. The conservative allowance can itself
be larger than the positive return credit.

For additive launch costs and a downward-closed feasible subset family,
an optimal subset cannot include a strictly negative individual drift score.
If resource `r` has minimum positive per-job cost `c_min,r`, no positive-cost
job launches when `Q_r > v h_max/c_min,r`. Hence, from empty queues,

\[
Q_r(T)\le v h_{\max}/c_{\min,r}+c_{\max,r},
\qquad
\sum_{t<T}c_r(x_t)\le T\bar c_r+Q_r(T).
\]

This proves bounded queues and a conditional expected-return credit bound
for the specified interface. It does not prove stationarity: all promises can
be zero while the true policy gradient is nonzero. It does not prove optimal
return relative to a different learner or HAPPO. A same-context certificate
maximization is not a competitive learning theorem.

## 4. The observable confidence bound fails before a trajectory is needed

For independent episode gradients `X_l` with `||X_l|| <= G` and common mean
`mu`, let `z=m^{-1}sum X_l`. Independence and Jensen give
`E||z-mu|| <= G/sqrt(m)`. Replacing one sample changes this norm by at most
`2G/m`. Bounded differences and a union bound over `M` adaptive but conditionally
frozen source batches therefore give the valid dimension-free radius

\[
r(m)=\frac{G}{\sqrt m}\left(1+\sqrt{2\log(M/\delta)}\right).
\]

For the explicit preflight `M=72`, `delta=0.05`, and `m=16`, this equals
**1.203441 G**. Since every possible empirical mean has `||z|| <= G`, the lower
gradient-norm bound `a_j` is identically zero, before adding any source motion
or staleness penalty. The launch promise and receipt certificate therefore
both maximize at zero step. This conclusion is algebraic and does not depend
on the A1 holdout outcomes.

Even with zero delay, the necessary episode count for `a_j > 0` at normalized
estimate norm `s=||z||/G` is

\[
m>\frac{(1+\sqrt{2\log(M/\delta)})^2}{s^2}.
\]

| Illustrative `s` | Necessary episode count | Current count |
|---:|---:|---:|
| 1.00 | 24 | 16 |
| 0.25 | 371 | 16 |
| 0.10 | 2,318 | 16 |

The illustrative norms are not measurements from A1. Positive norm confidence
is only necessary: future policy motion and the quadratic penalty can still
make the promised gain zero. A useful variance-dependent estimate may require
fewer episodes, but no such qualified estimate is currently supplied. Replacing
the certified `G` by an observed sample maximum is not valid without a tail bound.

The executable analytic report is
`results/launch_promise_certificate_feasibility_20261015.json`; it contains
21 parameter rows, the source hash, and zero scientific trajectories.

## 5. Novelty assessment

Delay-aware steps and convergence arguments already exist in
[Wu et al., ICML 2022](https://proceedings.mlr.press/v162/wu22g.html), including
asynchronous block-coordinate iterations. This directly limits a fixed-launch,
delay-adaptive-step novelty claim. Asynchronous SGD with virtual iterates and
delay-adaptive steps also has convex and nonconvex guarantees in
[Mishchenko et al., NeurIPS 2022](https://papers.nips.cc/paper_files/paper/2022/hash/029df12a9363313c3e41047844ecad94-Abstract-Conference.html).

These papers do not by themselves settle the interacting-policy MARL problem.
They do mean that a generic smooth-objective drift bound or delay-sensitive
scalar step is not enough to establish a distinctive ICML contribution.
The in-flight credit derivation above supplies an accounting interface, not a
verified first-of-kind claim. Its useful statistical instantiation and a
competitive return comparison are both absent at this decision point.

Fresh existence and metadata checks matched each official proceedings record
with its arXiv record. The machine-readable verification file records failed
or unqueried auxiliary resolvers rather than claiming they passed. No BibTeX or
manuscript reference list was edited.

## Authorized next boundary

The bounded successor-feasibility workflow terminates with the stated no-go.
No expensive RL qualification is launched because its score is already
provably degenerate under the tested sufficient bound. The transport correction
and conditional in-flight theorem remain reusable research artifacts.
Further development would need a new, **statistically useful return-gradient or
reference-relative estimator**, including its true sampling cost and a clear
distinction from existing delay-adaptive optimization. Merely changing the
experiment identifier, queue notation, or batch size does not meet that requirement.
