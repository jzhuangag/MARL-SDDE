# Paper-level decision: retain personalized learning, replace the safety-first mechanism

2026-09-01 (continuation began 2026-08-31); parent c84860732e24a9487216afb74fe4b380046d918c.
Internal AI-assisted research decision, not a manuscript or preregistration.

## Decision

**REVISE CORE; NO PERFORMANCE PILOT.** The current collection of checked
references is not an ICML-ready algorithm. Stop developing generic safe
parameter mixing as though only its confidence constants remained to be fixed.
Retain the scientific objective: improve each agent's own learning, measured
over its full declared state distribution and at matched total resource cost.

The selected next research candidate is **calibration-limited collaboration**:
transport or correct shared update information into the recipient's learning
problem, and determine whether the variance reduction survives the error,
delay and cost of learning that correction. This is a problem/mechanism
selection, NOT a claim that a new algorithm or a matching bound already exists.
Generic control variates and importance weighting are expressly inherited.

The single question is:

> Can the correction required for personalized collaboration be learned from
> the same delayed Markov data cheaply enough to produce a net finite-time
> learning advantage?

The central promise is no longer exact all-prefix no-harm under an unrestricted
unknown model. That was an increasingly strong design target, not a consequence
of the user's requested learning acceleration. A future paper may instead prove
a finite-time risk bound with an explicit calibration penalty and a sufficient
net-benefit condition. This is an OPENLY PROPOSED change of future claim, not a
retroactive relaxation of T-083A or any previous experimental gate.

## 1. Why the previous routes are not being combined into a paper

| Route | What is actually available | Decision |
|---|---|---|
| Exact future-risk transfer | Correct Markov-jump risk identity; observed-data references need expensive sensing/reset access | Keep as an oracle/reference, not the affordable algorithm |
| Compressed absolute-tail transfer | Valid bounds; declared diagnostic rejected every useful direction even without sampling uncertainty | Stop this implemented shield, preserve the counterevidence |
| Delayed executed-risk ledger | Cumulative visited-state comparison under stated conditional-law assumptions | Keep the restricted theorem; do not relabel it as full-state MSE |
| Generic bias/importance correction | Strong existing personalized-learning methods | Required baselines, not sufficient new contributions |
| Learning the correction at matched cost | A precise unresolved mechanism and resource question | Selected for ONE complete theoretical feasibility decision |

The first three rows are supported by the local notes
`rl_collaboration_integration_decision.md`, `td_contraction_feasibility_decision.md`
and `delayed_training_risk_decision.md`. They do not prove that all collaboration
is impossible. They do show why another wrapper or a larger experiment count
would not close the current argument.

## 2. A newly identified close baseline changes the novelty bar

AffPCL [1] already combines personalized bias and importance correction,
affinity-dependent rates, asynchronous estimation, and a multi-variable
Lyapunov argument. Its TD discussion explicitly uses independent draws from
an offline dataset (C.6, footnote 5). Section 6.1 and Theorem 2 address the
difficulty of estimating unknown density ratios. Its rate comparison is not
our realized all-prefix safety claim. These distinctions are specific read
boundaries, not a claim that its complete theory has been independently audited.

SCAFFLSA [2] already corrects heterogeneity bias using control variates for
federated linear stochastic approximation and TD. Its stated global averaged
linear-system target differs from fully personalized targets; its observation
model also separates independent clients from within-client Markov dependence.
Importance-weighted source-transition transfer with a finite-sample batch-RL
analysis is older still [3].

Consequently, neither “personalization + adaptive collaboration + Lyapunov”
nor “replace parameter averaging with weighted samples” passes novelty alone.
Adding a mixing factor or bounded-delay term to an inherited rate is not the
selected contribution. A defensible advance must solve an attainable
calibration-and-learning tradeoff that the chosen comparison leaves unresolved.
The present bounded search does not establish priority for that advance.

## 3. Frozen-for-reasoning scope, not frozen experimental settings

Start with finite, stationary, fixed-policy Markov reward processes, one per
recipient. Each has its own transition law and target value. Use tabular
prediction initially, so a projected Bellman fixed point is not silently
substituted for the true value. Declare a strictly positive state weighting
nu_i BEFORE outcomes. The primary learning objective is

\[
J_T=\frac1N\sum_{i=1}^N\sum_{t<T}
  \mathbb E\|v_{i,t}-V_i\|_{\operatorname{diag}(\nu_i)}^2.
\]

Endpoint risk and per-agent costs are additional outcomes, not replacements
for J_T after inspecting results. A guarantee against local training does not
establish superiority over a strong static collaborative learner.

Training is coordinated through timestamped messages; recipients keep their
own parameters and goals. Environment policies are fixed during the initial
theorem. No execution-time inter-agent communication, POMDP inference or
joint-action MARL guarantee is being introduced. Common representations and
the exact transferable records must be explicit. Receiving another agent's
reward does not give access to one's own reward on that transition.

The proposed change is from transporting whole parameter vectors to
transporting recipient-corrected UPDATE INFORMATION. Known reward relabeling,
unknown reward models, conditional transition probabilities, raw-data sharing
and reset access are different interfaces: the next decision must choose one,
price its observations, and cannot import the convenient parts of all of them.
Initially independent environment innovations conditional on the joint past
are a permissible restriction. Cross-agent common noise is not automatically
covered by per-agent Markov assumptions.

## 4. The mathematical obstruction is conditional calibration, not just similarity

A stationary importance identity for a fixed function f does not imply the
same identity conditional on the current training history for adaptive f_t.
Consider two independent chains with the SAME kernel

\[
P=\begin{pmatrix}.9&.1\\.1&.9\end{pmatrix}.
\]

Their stationary state and transition-tuple distributions coincide. Condition
on the recipient currently being in state 0 and the donor in state 1. For
f(y)=1{y=1}, the recipient's next-state expectation is .1; an equally pooled
next-state sample has expectation .5. Stationary ratios equal one and do not
repair this CONDITIONAL discrepancy. The conditional mixture has probability
.5 on each state; weighting its next state by P(0,y)/.5 gives expectation .1.

This is an exact identity check with known probabilities, not an observable
algorithm or evidence that AffPCL fails. Markov stochastic approximation can
converge without one-step conditional unbiasedness, but its dependence terms
need an actual argument. The example rules out importing an i.i.d. cancellation
proof unchanged. Estimating the conditional ratio also has a cost; knowing it
for this arithmetic check does not make it available to the learner.

Delay adds a second timing issue. A received old sample is already part of the
current history; it cannot be called a fresh centered innovation at delivery.
The next algorithm must use a declared birth-time filtration, a valid stale
update analysis or a proved predictive correction. Merely attaching a
timestamp, or separating two adjacent blocks, does not supply independence.

## 5. What Lyapunov must prove in the revised candidate

The following is a standard algebraic interface, NOT a completed Markov-TD
theorem or a novelty claim. For one recipient write its actual error recursion
as e_(t+1)=F e_t+eta(b_t+xi_t), where F is stable, b_t is the full conditional
mean discrepancy, and E[xi_t|F_t]=0. For M positive definite define

\[
P_L=\sum_{k\ge0}(F^k)^\top M F^k,
\qquad P_L-F^\top P_LF=M,\qquad L(e)=e^\top P_Le.
\]

P_L is an analysis metric unless an affordable estimator is separately given;
this definition does not authorize a matrix inverse in an online algorithm.
Expanding the quadratic and taking conditional expectations yields exactly

\[
\mathbb E[\Delta L\mid\mathcal F_t]
=-e_t^\top M e_t
 +2\eta e_t^\top F^\top P_L b_t
 +\eta^2 b_t^\top P_L b_t
 +\eta^2\operatorname{tr}(P_L C_t),
\quad C_t=\mathbb E[\xi_t\xi_t^\top\mid\mathcal F_t].
\]

This links the drift to FULL parameter/value risk, not a retrospective queue.
Summing gives a finite-time identity; bounding every discrepancy is the work.
In actual Markov TD, b_t includes state-conditioned versus stationary operator
effects, estimated corrections and stale parameters. Defining xi_t to have
zero mean does not make b_t small. A Poisson-equation/block argument and a
coupled calibration-error recursion are still required.

If a truly centered auxiliary innovation z is available, correcting local
noise epsilon by Wz changes its conditional covariance to

\[
\Sigma_\epsilon-C_{\epsilon z}W^\top-W C_{\epsilon z}^\top
                       +W\Sigma_zW^\top.
\]

That is the ordinary control-variate identity. The desired algorithm must
earn the reduction in this term while controlling the added b_t terms and
resource cost. A composite Lyapunov function may include calibrated-model
errors and delay-memory terms ONLY after their update recursions are specified
and their drift inequalities derived. Queue terminology cannot replace this.
SDDE is not needed for this discrete interface; no approximation theorem is
claimed.

### A sharp scalar check: variance reduction can lose to calibration bias

For e_(t+1)=a e_t+eta(b+epsilon_t), with mu>0, eta>0, constant b,
a=1-eta*mu in (-1,1), and zero-mean stationary AR(1) noise of variance
sigma^2 and coefficient |lambda|<1 with independent innovations, the
steady-state MSE is

\[
\frac{b^2}{\mu^2}+
\frac{\eta^2\sigma^2(1+a\lambda)}{(1-a^2)(1-a\lambda)}.
\]

To derive it, remove the stationary mean b/mu. The cross moment solves
c= a*lambda*c+eta*lambda*sigma^2, and the variance solves
v=a^2*v+2*a*eta*c+eta^2*sigma^2. These two equations give the expression.
This is an infinite-time diagnostic, not the finite-T guarantee promised above.

For deterministic e_0 and stationary noise at initialization, the finite-time
counterpart is also explicit:

\[
\mathbb E e_t^2=
\left(a^t e_0+\frac b\mu(1-a^t)\right)^2+
\eta^2\sigma^2\sum_{k=0}^{t-1}\sum_{l=0}^{t-1}
 a^{k+l}\lambda^{|k-l|}.
\]

It follows by unrolling the recursion and using the full noise covariance,
not by replacing the chain with i.i.d. noise. Sum over the same pre-update
times as J_T for a cumulative comparison. If calibration leaves H rather
than T learning updates at the same resource deadline, evaluate endpoint risk
at H for that algorithm and T for a baseline that did not pay that cost.
For CUMULATIVE risk, use the same declared resource-time clock and include
the algorithm's predictions during calibration/holding intervals; comparing
J_H to J_T while omitting those intervals would change the estimand. A random
warm-up state correlated with a learned b requires its additional joint
moments; this formula does not erase them.

With mu=1, eta=.1, lambda=0, local variance 1 and corrected variance .5,
local MSE is .05263158 and perfectly centered corrected MSE is .02631579.
A residual bias .1 gives .03631579, but bias .2 gives .06631579 and loses.
Net benefit requires |b|<.16222142. At the same lag-zero variance 1, changing
lambda from 0 to .8 changes stationary MSE to .32330827. Thus neither
one-step variance reduction nor a scalar similarity score proves learning gain.

These are deterministic arithmetic examples, not new scientific trajectories,
threshold tuning or evidence that an implementable correction attains those
noise/bias pairs. The comparison must also include fewer available updates
if calibration consumes additional resources. Warm-up estimates reused in
training are generally correlated with the learner and cannot be declared
independent to simplify that comparison.

## 6. One contribution contract and one remaining pre-experiment decision

The prospective story is: useful collaboration is limited by how accurately
and cheaply each learner can translate shared information into its own update
law. The positive contribution, IF achieved, is a single mechanism that learns
this translation and realizes net acceleration; its calibration boundary and
Lyapunov risk bound explain the same phenomenon. It is not five separate
contributions involving a graph, queue, schedule, delay and step size.

Candidate assessment (judgment scores, not acceptance probabilities):

| Candidate | Feasible | Interesting | Novel now | Ethical | Relevant | Disposition |
|---|---:|---:|---:|---:|---:|---|
| Exact certified future-transfer wrappers | 2 | 4 | 3 | 5 | 4 | Implemented cost/activation gate fails |
| Visited-risk ledger as the paper | 5 | 2 | 1 | 5 | 3 | Reject as final mechanism |
| Generic personalized control variates | 4 | 4 | 1 | 5 | 4 | Baseline, not novelty |
| Learned correction with end-to-end Markov/resource cost | 3 | 4 | 2 | 5 | 4 | Research candidate only |

Novelty remains unqualified despite the last row's potential. The NEXT bounded
deliverable is one fully specified finite-MRP calibration/learning construction,
not another literature-only roadmap. It must include all of:

1. Actual observable records, reward interface, birth/delivery timing and local
   parameter update. No oracle values, unexplained density-ratio oracle or free
   reward queries; include all calibration storage, arithmetic and messages.
2. An attainable finite-data calibration bound for the exact same stream,
   including adaptive reuse. Conditional bias, Markov dependence and delay
   must enter the actual Lyapunov recursion, not just a proposed theorem title.
3. A proved nonempty benefit condition AFTER the cost of calibration, relative
   to actual local training. An assumed variance reduction is not this proof.
4. A precise distinction against AffPCL and importance-weighted transfer,
   including its DRE assumptions; a routine Markov extension alone FAILS.
5. A single analytic/CPU feasibility protocol that could falsify the complete
   construction. Only if 1-4 pass may it be independently preregistered and run.

If this construction only works with privileged true models, vacuous bounds,
unpriced sensing, or duplicates the baseline, reject it and report that no
qualified successor has been found. Do not respond by allocating another
experiment identifier or silently selecting a more convenient risk metric.

Any later efficacy comparison must include independent local training,
appropriate corrected-update/AffPCL baselines, strong static collaboration,
and matched short-lookahead alternatives where applicable. Each baseline owns
its complete training history and receives the same stated information and
resource accounting. Start stationary and genuinely temporally correlated;
nonstationarity and neural control are not required before this gate closes.

## 7. Evidence and execution boundary

T-083A remains its frozen FAIL, including F12 and F13. Its positive affine
mechanism observations do not validate a new corrected-update method.
No frozen controller, gate, seed or result is changed by this decision.
No new performance trajectory, pilot/formal seed, GPU/HPC4 access or remote
storage operation is authorized here. Full regression checks implementation,
not the existence of an excellent paper. There is no ICML acceptance estimate.

This completes the requested paper-level decision rather than claiming the
paper is complete: the safety-first architecture is not the final mainline;
the original learning goal remains; the successor has a concrete mechanism
question and an explicit unresolved novelty/feasibility gate.

## Verified sources (bounded reading)

[1] Chenyu Zhang and Navid Azizan. Personalized Collaborative Learning with
Affinity-Based Variance Reduction. ICLR 2026; arXiv:2510.16232v3.
[Primary record](https://arxiv.org/abs/2510.16232).

[2] Paul Mangold, Sergey Samsonov, Safwan Labbi, Ilya Levin, Reda Alami,
Alexey Naumov and Eric Moulines. SCAFFLSA: Taming Heterogeneity in Federated
Linear Stochastic Approximation and TD Learning. arXiv:2402.04114v3.
[Primary record](https://arxiv.org/abs/2402.04114).

[3] Andrea Tirinzoni, Andrea Sessa, Matteo Pirotta and Marcello Restelli.
Importance Weighted Transfer of Samples in Reinforcement Learning.
ICML 2018, PMLR 80:4936-4945.
[Proceedings](https://proceedings.mlr.press/v80/tirinzoni18a.html).

Fresh existence/metadata checks and exact read scopes are preserved in
`collaboration_paper_level_sources.json`. This note is internal research
planning; no manuscript or bibliography is delivered.
