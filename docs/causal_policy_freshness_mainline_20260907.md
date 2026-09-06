## Material Passport

- Origin Skills: academic-research-suite; publishable-academic-writing
- Origin Mode: research synthesis
- Origin Date: 2026-09-07
- Verification Status: VERIFIED AS CURRENT RESEARCH THESIS
- Version Label: causal_policy_freshness_mainline_v1

# Causal Policy Freshness: the ICML 2027 research thesis

## Candidate title

**Causal Policy Freshness: Lyapunov Scheduling of Policy-Cache Graphs for
Communication-Efficient Asynchronous MARL**

## Thesis

Asynchronous centralized training with decentralized execution (CTDE) creates
a freshness problem that is specific to multi-agent reinforcement learning.
A rollout worker owned by agent `i` uses the current owner policy together with
cached versions of its teammates' policies.  Refreshing every teammate before
every rollout eliminates this mismatch but incurs complete-graph
communication; refreshing by age alone ignores that stale teammate policies
can change the owner gradient in either direction.

The proposed method treats policy freshness as a signed, causal control
problem.  A launch-state causal cone identifies which teammate policies can
affect the next finite-horizon owner update.  Within this cone, a centralized
critic estimates the signed learning drift created by each candidate cache
refresh.  A Lyapunov drift-plus-penalty rule selects the null action or one
directed refresh under a long-run message budget.  Randomly delayed packets are
paired with their launch decisions in the analysis.  The result is a dynamic
training-time policy-cache graph; decentralized execution is unchanged.

This is one mechanism and one claim: **spend asynchronous synchronization only
where a fresh teammate policy has certified marginal learning value.**

## System model

There are distinct policy blocks

\[
\theta=(\theta_1,\ldots,\theta_n)
\]

in a cooperative Markov game with potential loss `F(theta)`.  A worker owned
by block `i` maintains

\[
\chi_i=(\theta_i,\chi_{1\to i},\ldots,\chi_{n\to i}).
\]

The owner block is synchronized as a base operation.  Optional edge
`j -> i` transmits the current `theta_j` to that worker before its next rollout;
it never copies `theta_j` into `theta_i`.  A returned trajectory updates only
its owner block, while packets for other owners can be in flight.  This is
asynchronous block-coordinate policy learning, not homogeneous data-parallel
workers updating one shared actor.

Training is centralized: the learner has the critic, policy-version ledger,
worker caches, packet birth times, and communication queue.  Execution is
decentralized: each final actor uses its local observation, with no policy-cache
graph or additional execution-time message.

The primary experimental estimands are learning return/risk at matched actor
transitions and policy bytes.  Wall-clock time and scheduler overhead are
reported as secondary systems quantities.

## The Lyapunov decision

At launch `p`, the public dynamics and current environment state define an
eligible causal cone `C_p` and action set

\[
\mathcal A_p=\{\varnothing\}\cup
\{\{j\to i_p\}:j\in\mathcal C_p\}.
\]

For action `a`, let `mu_p(a)` and `m_p(a)` be the conditional mean and second
moment of the future owner-gradient packet, `g_p^0` the current owner-block
potential gradient, `w_p` the fixed stable receipt step, and `d_p(a)` a
predictable launch-to-receipt gradient-motion bound.  The paired learning-drift
upper bound is

\[
D_p(a)=
-w_p\langle g_p^0,\mu_p(a)\rangle
+\frac{L_{i_p}w_p^2}{2}m_p(a)
+w_pd_p(a)\|\mu_p(a)\|.
\]

The first term makes the graph signed: freshness is valuable when it improves
alignment with the current learning direction, not simply when it removes a
large parameter mismatch.

The learning potential is augmented by the exact policy-cache energy

\[
H_p=\frac12\sum_{i\ne j}\beta_{ji}
\|\theta_{j,p}-\chi_{j\to i,p}\|^2.
\]

Refreshing `j -> i` decreases this energy by the observable amount
`B_p(j->i)=beta_(ji)||theta_j-chi_(j->i)||^2/2`.  An owner update creates an
exact outgoing-cache increment, retained as a receipt-time remainder or a
predictable action-dependent bound.  This term prevents an uncertain critic
from making the null graph absorbing while still charging every refresh.

Let `Q_p` be the virtual communication queue,

\[
Q_{p+1}=[Q_p+c_p(a_p)-\bar c]^+.
\]

Given a predictable critic/Jacobian-vector-product estimate `Dhat_p`, the
executed action is

\[
a_p\in\arg\min_{a\in\mathcal A_p}
\{V\widehat D_p(a)-B_p(a)+Q_pc_p(a)\}.
\]

This is the one-step drift of `V F+H+Q^2/2`: Lyapunov drift therefore
determines the communication graph online; it is not only a post-hoc
convergence tool.  Null-plus-one-edge selection is exact in
`O(Delta_p)` after the local signed statistics are formed.  The primary
algorithm fixes rollout horizon and receipt step so that the paper studies one
identifiable control variable rather than combining previously unsupported
horizon and step-size heuristics.

## Main theorem target

Assume (a) block smoothness and a lower-bounded potential; (b) eventual packet
receipt with an explicit in-flight bound; (c) a uniformly geometrically mixing
trajectory kernel or regenerative alternative; (d) a predictable
launch-to-receipt motion bound; (e) bounded packet second moments; and (f) a
simultaneous expected learning-drift error `epsilon_p^F` over the finite action
set.

The paired launch-receipt theorem gives

\[
\sum_{p<N}\kappa_p\mathbb E\|g_p^0\|^2
\le F(\theta^0)-F_\star+
\frac{H_0+Q_0^2/2}{V}+
\sum_{p<N}\mathbb E R_p+
\frac1V\sum_{p<N}\mathbb E\Gamma_p^++
2\sum_{p<N}\mathbb E\epsilon_p^F+
\frac{NB_Q}{V},
\]

against any launch-measurable randomized comparator whose conditional expected
cost is at most the budget rate and which satisfies the declared descent
condition.  This conditional formulation permits non-null unit-cost actions
when the budget rate is fractional; a merely pathwise average-feasible
comparator would not remove the queue cross term.  Queue iteration gives the
corresponding average-message bound.  With bounded stochastic packet variance, `w=N^{-1/3}` and
`V=N^{2/3}` balance the stationarity and budget terms at order
`N^{-1/3}`, apart from normalized Markov, motion, and score-estimation terms.

The favorable phase is explicit: refresh has learning value when the reduction
in mixed-policy bias is large enough to dominate packet variance,
launch-to-receipt motion, and its queue price.  This phase statement explains
why dense or weak-interaction games may rationally choose the null or full
baseline without supporting a universal-dominance claim.

## Estimation path

The exact affine model supplies an observable system-identification corollary.
Every completed packet reveals a noisy projection of its unknown reference;
translation-equivariant normalized least mean squares contracts under the
cyclic-owner persistent-excitation condition.  Its reference error yields an
explicit learning-drift error `epsilon_p^F`, which enters the theorem above
through the twice-uniform-error action-regret term.

For nonlinear MARL, the learner uses completed replay data and its centralized
critic.  One local reverse-mode differentiation estimates how the owner
gradient alignment changes along each known policy-version displacement in the
causal cone.  The structural derivative support is `O(Delta)`; actual runtime
and memory are measured rather than inferred from asymptotic notation.

## Evidence chain

The current evidence supports consecutive links of the same mechanism:

1. **Causal locality on a standard task.**  PDSG-CONE-001 established calibrated,
   state-dependent short-horizon policy-influence cones on 20-agent
   Pistonball; the cone expands with rollout horizon.
2. **Signed graph value.**  PDSG-SIGN-001 showed that exact signed
   Lyapunov scheduling outperforms a 24-policy feasible envelope under random
   delay and binding messages.
3. **Observable recovery.**  PDSG-OBS-001 used only completed training packets,
   improved terminal risk by 58.99% and cumulative risk by 5.40% in all 16
   active exact-model cells, and recovered 93.67% median oracle headroom with
   byte-identical reproduction.
4. **Sparse score feasibility.**  PDSG-EST-001 established favorable-regime
   signed ranking and a finite-action expected-maximum error bound for the
   local critic/JVP interface.

An outcome-free CPU contract now connects this formula to 20 distinct
Pistonball actors and a centralized critic: all eligible edge derivatives are
formed with one cross-policy VJP, the update changes only the owner block, and
cache refreshes are charged by exact actor payload bytes.  The remaining
empirical link is end-to-end Pistonball learning.  That experiment decides
whether the complete method, rather than its individual interfaces, improves
the return--communication frontier.

The first deterministic GPU development matrix validates the complete systems
path but not efficacy.  The composite scheduler changes the mixed-policy
training trajectory and uses 984 of 2,048 refresh units, yet all methods tie at
the terminal decentralized return and the signed-only ablation selects no
edge.  The actor--critic backbone must first pass a separate learning
qualification; these development outcomes do not authorize a pilot.

## Distinction from adjacent work

- [IMPALA](https://proceedings.mlr.press/v80/espeholt18a.html) corrects
  actor--learner lag for one shared policy; the present graph selectively
  refreshes teammate blocks in distinct interacting policies.
- [Min et al.](https://proceedings.mlr.press/v202/min23a.html) study
  asynchronous communication among homogeneous agents solving one linear MDP;
  the present action controls mixed joint-policy snapshots inside one Markov
  game.
- [Agent-Centric Actor-Critic](https://proceedings.mlr.press/v267/jung25a.html)
  addresses execution-time macro-action asynchrony; the present asynchrony is
  training-time launch and receipt of policy-gradient packets.
- [Dynamic Coordination Graph](https://proceedings.mlr.press/v157/siu21a.html)
  generates a graph for value factorization and action inference; the present
  graph is a resource-constrained cache-refresh action during training.
- [CAIC](https://proceedings.mlr.press/v337/li26h.html) schedules delayed
  intent messages used by agents during execution; the present messages are
  policy versions used by rollout workers during centralized training.
- Yu, Chen, and Poor's SDDE framework optimizes communication and computation
  for delayed workers updating a common stochastic-optimization parameter; the
  present state contains distinct policy blocks and directed teammate caches,
  and the online action is a signed policy-dependency edge.

The distinction is supplied by the complete problem--algorithm--theorem
combination, not by the word “graph,” “delay,” or “Lyapunov” separately.

## Role of the SDDE

The exact event-time recursion is the primary model and proof.  Under a
small-step scaling, parameters, directed caches, packet histories, and the
communication queue admit a controlled hybrid stochastic delay differential
equation with cache-reset jumps.  Functional Itô--Dynkin analysis can expose
how delay, interaction sparsity, and queue price move the favorable-phase
boundary.

The SDDE belongs in the appendix only if a finite-horizon weak-convergence or
coupling theorem connects it to the discrete process.  Otherwise the paper
retains the exact paired Lyapunov theorem and omits an unsupported diffusion
claim.  This keeps stochastic optimization central without making the story
depend on an unnecessary approximation.

## Paper-level experiment design

The main positive benchmark is 20-agent Pistonball with distinct actor blocks,
fixed horizon, heterogeneous service delay, and matched transition/policy-byte
budgets.  The principal plot is return versus optional policy bytes, with
sample-progress curves and tail performance.  Comparators are no optional
refresh, complete refresh, age and mismatch scheduling, physical static graph,
best fixed local graph/rate, and the strongest resource-feasible envelope.

A dense cooperative task serves as the boundary condition.  Ablations remove
the signed score, causal cone, and queue separately.  Additional panels vary
delay, message budget, number of agents, and local interaction degree, and
report estimator overhead and graph turnover.

The central benchmark gate is a broad improvement in return/sample efficiency
over the strongest matched non-oracle scheduler while satisfying the message
budget.  Wall-clock improvement is supportive rather than necessary; excessive
critic/JVP overhead remains a practical failure mode and is reported directly.

## Immediate execution order

1. freeze the complete Pistonball owner-update, critic, packet, and accounting
   contract around the validated neural interface;
2. specify the replay-only critic/VJP uncertainty and regularity assumption;
3. preregister a small GPU Pistonball pilot with new seeds and mandatory
   strong-baseline gates;
4. run formal seeds and write the full manuscript only if that pilot passes.

This sequence advances the one surviving thesis and does not reopen the
discarded participation, unsigned graph, online-horizon, or generic
wall-clock-recovery routes.
