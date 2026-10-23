## Material Passport

- Origin Skills: academic-research-suite; publishable-academic-writing
- Origin Mode: research synthesis
- Origin Date: 2026-09-07
- Verification Status: CONDITIONAL RESEARCH THESIS; EXECUTABLE CONTROLLER NOT YET VALIDATED
- Version Label: causal_policy_freshness_mainline_v1

# Causal Policy Freshness: the ICML 2027 research thesis

## Evidence status as of 2026-09-09

This remains a candidate research question, not a completed contribution.
MW-PF-DEV-001 permanently rejected the exact cache-energy term as a sufficient
continuation-value proxy: every cross-fitted fold selected zero reset weight
and the candidate failed four mandatory learning-value gates.  The positive
exact dynamic-oracle ceiling therefore cannot be presented as evidence for the
proposed online controller.  The only current successor is the separately
documented drift-structured residual-value formulation, whose learning,
confidence, and MARL convergence guarantees remain open.  No standard-MARL or
GPU efficacy experiment is authorized by the present evidence.

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

This is one mechanism and one claim: **use the learning Lyapunov drift to spend
asynchronous synchronization on the teammate factor, and at the packet weight,
with the best predicted finite-horizon learning value under the current dual
resource price.**

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

For action `a`, let `A_p(a)` be the conditional alignment of the future
owner-gradient packet with the launch-time potential gradient, `G_p(a)` an
action-specific packet-gradient bound, and `M_p` a predictable
launch-to-receipt parameter-motion bound. Combining block smoothness with the
topology-robust core Lyapunov function `VF+Q^2/(2 nu)` gives the principal
action-dependent index

\[
J_p^{\rm core}(a,\alpha)=-m_p^{\rm core}(a)\alpha
+\frac{C_p^{\rm core}(a)}2\alpha^2+Q_pc_p(a),
\]

where

\[
m_p^{\rm core}(a)=V[A_p(a)-L_{i_p}G_p(a)M_p],
\qquad C_p^{\rm core}(a)=VL_{i_p}G_p(a)^2.
\]

The first term makes the graph signed: freshness is valuable when it improves
alignment with the current learning direction, not simply when it removes a
large parameter mismatch.

For an optional persistent-cache strengthening, fix one edge universe `U` and
augment the learning potential by the exact policy-cache energy

\[
H_p=\frac12\sum_{(j,i)\in U}\beta_{ji}
\|\theta_{j,p}-\chi_{j\to i,p}\|^2.
\]

Refreshing `j -> i` decreases this energy by the observable amount
`B_p(j->i)=beta_(ji)||theta_j-chi_(j->i)||^2/2`.  An owner update creates an
exact outgoing-cache increment, retained as a receipt-time remainder or a
predictable action-dependent bound. State-dependent candidate support may
change inside `U` without changing `H`. If the weights or universe change, the
exact topology-motion jump must be added; summing only over currently active
edges is not a valid telescoping argument.

Let `Q_p` be the virtual communication queue,

\[
 Q_{p+1}=[Q_p+\nu(c_p(a_p)-\bar c)]^+.
\]

The queue contribution to the composite Lyapunov function is
`Q_p^2/(2 nu)`, so this scaled recursion preserves the action price `Q_p c_p`
while making `nu` an explicit dual-response step rather than an untracked
implementation coefficient.

Given a predictable critic/Jacobian-vector-product estimate `Ahat_p`, the
executed action and packet weight are

\[
\widehat\alpha_p(a)=
\Pi_{[0,\bar\alpha]}\!\left(
\frac{\widehat m_p^{\rm core}(a)}{C_p^{\rm core}(a)}\right),
\qquad
a_p\in\arg\min_{a\in\mathcal A_p}
J_p^{\rm core}(a,\widehat\alpha_p(a);\widehat A_p).
\]

The principal rule is the one-step drift of `V F+Q^2/(2 nu)`: Lyapunov drift therefore
determines the communication graph and receipt weight online; it is not only a
post-hoc convergence tool.  Null-plus-one-edge selection and its scalar weight
are exact in `O(Delta_p)` after the local signed statistics are formed.  The
rollout horizon is fixed, so the two controlled quantities have a single
interpretation: which teammate cache enters the next owner trajectory, and how
strongly its delayed packet is applied. The cache-augmented rule replaces the
core coefficients by fixed-universe cache coefficients and is analyzed as a
separate extension.

## Main theorem target

Assume (a) block smoothness and a lower-bounded potential; (b) eventual packet
receipt with an explicit in-flight bound; (c) a uniformly geometrically mixing
trajectory kernel or regenerative alternative; (d) a predictable
launch-to-receipt motion bound; (e) bounded packet second moments; and (f) a
simultaneous expected learning-drift error `epsilon_p^F` over the finite action
set.

The topology-robust paired launch-receipt theorem gives

\[
\sum_{p<N}\kappa_p\mathbb E\|g_p^0\|^2
\le F(\theta^0)-F_\star+
\frac{Q_0^2/(2\nu)}{V}+
\sum_{p<N}\mathbb E R_p+
2\bar\alpha\sum_{p<N}\mathbb E\epsilon_p^F+
\frac{NB_Q}{V},
\]

against any launch-measurable randomized comparator whose conditional expected
cost is at most the budget rate and which satisfies the declared descent
condition.  This conditional formulation permits non-null unit-cost actions
when the budget rate is fractional; a merely pathwise average-feasible
comparator would not remove the queue cross term.  Queue iteration gives the
corresponding average-message bound. The displayed left side is launch-time
selected-block stationarity. If owners are drawn from predictable
probabilities bounded below by `pi_min`, a tower-property corollary converts it
to full-gradient stationarity at a uniformly sampled launch, at the expected
`1/pi_min` factor. A cyclic implementation instead needs an epoch-motion
argument. With bounded stochastic packet variance, `w=N^{-1/3}` and
`V=N^{2/3}` balance the stationarity and budget terms at order
`N^{-1/3}`, apart from normalized Markov, motion, and score-estimation terms.

This core bound has no graph-turnover term. The optional fixed-universe cache
corollary adds `H_0/V`; if cache weights vary, it also adds the positive part of
the exact topology-motion increment. Pursuit turnover therefore cannot be
advertised empirically while disappearing theoretically.

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
edge.  A separately frozen backbone qualification then passes: under fully
fresh caches, actor step `0.01`, and one development-only seed, terminal return
improves by 4.826 from a common initialization, with finite nonzero gradient
and actor-drift diagnostics.  This retains the distinct-actor
centralized-critic learner but is not a communication comparison: complete
freshness costs 77,501 refresh units, and no matched optimizer control was part
of that learner-only gate.  The next evidence link is therefore a matched,
binding-budget controller-headroom experiment, not an efficacy claim.

The first outcome-blind score-scale audit exposed a sharper implementation
condition before that experiment: the ReLU centralized critic has zero mixed
action curvature almost everywhere, so all 980 measured cross-policy signed
VJP deltas were exactly zero. Cache-reset and queue terms remained active.
This stops the ReLU score interface and explains the historical signed-only
null behavior. The next allowed learner uses a fixed smooth critic action
head and must requalify both nonzero mixed VJPs and learning before controller
headroom is measured.

That smooth-interface qualification now passes. A fully fresh SiLU learner
improves terminal decentralized return by 2.1068 under its frozen gate, and a
matched scale run produces nonzero signed deltas with 42.77% favorable-edge
frequency while both cache and queue prices remain active. These are
development interface checks, not controller evidence. The outcome-blind
normalization fixes `V=1e8` and `beta=83,650.1231` for the next matched
headroom design; signed-only and cache-only ablations remain mandatory because
the signed raw curvature is small.

That matched two-seed design has now failed its frozen gate. The proposed
one-VJP controller trails no-refresh in both seeds, full refresh has opposite
paired effects (`-12.59` and `+1.13` relative to no-refresh), and the cache
term degrades signed-only. The result rules out the current linearized
finite-displacement score and cache weighting; it does not support a pilot.
It also makes the scientific problem sharper: staleness can stabilize or harm
learning, so mismatch minimization is not a valid objective by itself.

The paired Lyapunov theorem is estimator-agnostic. The only remaining
Pistonball repair allowed before stopping this benchmark line is an exact
sparse counterfactual score: evaluate owner-gradient alignment under each
actual one-edge cache replacement in the causal cone, with measured
`O(Delta)` reverse-mode cost. This changes the estimator interface, not the
budget, learner, or claim. It must pass an outcome-free complexity audit and a
new frozen headroom gate; otherwise the standard-benchmark mainline stops.

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

Pursuit was the first positive benchmark candidate because its public local
observations admit an outcome-free degree-four state-dependent interaction
interface.  It uses distinct actor blocks, fixed horizon, heterogeneous
service delay, and matched transition/policy-byte budgets.  The intended
principal plot was return versus optional policy bytes, with sample-progress
curves and tail performance.
Comparators are no optional refresh, complete refresh, age and mismatch
scheduling, a fixed local graph, best fixed local graph/rate, and the strongest
resource-feasible envelope.  This experiment remains conditional on a neural
score-interface qualification; structural sparsity alone is insufficient.

Pistonball is retained as the dense/high-degree degeneration case rather than
the positive benchmark.  Ablations remove the signed score, state-dependent
factor graph, packet-weight control, and communication queue separately.
Additional panels vary delay, message budget, number of agents, and local
interaction degree, and report estimator overhead and graph turnover.

The central benchmark gate is a broad improvement in return/sample efficiency
over the strongest matched non-oracle scheduler while satisfying the message
budget.  Wall-clock improvement is supportive rather than necessary; excessive
critic/JVP overhead remains a practical failure mode and is reported directly.

## Final benchmark decision (2026-09-07)

The exact sparse counterfactual score audit at commit `8962d1a` failed its
pre-outcome complexity gate. The score was nonzero and active, and its measured
runtime ratio to no refresh was 1.375, but the registered causal cone required
9--18 reverse evaluations rather than at most eight. Thus standard Pistonball
does not instantiate the assumed low-degree policy-dependency factorization.

Per the frozen decision rule, there will be no further Pistonball headroom
matrix, pilot, or formal run for this mainline. Pruning the observed cone,
relaxing the gate, or replacing the score after seeing this result would be a
new method rather than a validation. The conditional discrete Lyapunov theory
and the exact causal estimator remain useful research artifacts, but the
present problem--benchmark--algorithm package is not an ICML-ready positive
paper. Any continuation must begin with a new formulation-level feasibility
argument and outcome-free headroom certificate, not another experiment number
on this benchmark.

## Locally factored continuation (2026-09-07)

The continuation makes local factorization part of the problem statement
rather than inferring it after an experiment.  The outcome-free Pursuit audit
found a state-dependent degree-four interface with negligible cap truncation
and substantial graph turnover.  The joint Lyapunov rule now exactly selects
both a null/one-edge cache refresh and its receipt-time packet weight in
`O(Delta)` candidate evaluations.  A finite factor-switch Markov game proves a
25.18% equal-communication advantage over the strongest fixed-initial-state
and round-robin schedulers.

The stochastic interface has also narrowed.  Finite-state Poisson holdout
bounds retain Markov transients and policy-induced occupancy shift, while a
count-uniform rectangular robust dynamic program preserves the current local
state.  Independent MCERT-001 confirmation selected the correct
state-compatible edge in every primary case and recovered 53.98% median and
49.65% fifth-percentile exact value after 8,192 fully charged identification
transitions; the incompatible edge was never certified positive.  This is a
positive theorem-interface result, not standard-task return evidence.

The end-to-end CPU tabular mechanism link is now closed by PDSG-FR-001 on 32
untouched seeds. Its core
`VF+Q^2/(2nu)` controller reduces favorable-phase geometric cumulative risk by
41.22% relative to the frozen strong state-myopic scheduler, improves all
12/12 favorable cells and 384/384 paired seed-cells, and recovers 87.57% median
oracle headroom. Primary and isolated-reproduction endpoints and summaries are
byte-identical. This is finite-state core-mechanism evidence only: the runner
contains no persistent cache energy and does not certify a neural Pursuit
critic.

This result authorizes an outcome-free Pursuit cache/critic interface audit,
not a GPU efficacy run. No claim relies on reopening the stopped Pistonball
line.

The outcome-free cache interface has now passed on actual Pursuit state
sequences. Launch refreshes charge full actor bytes and persist independently
of a zero receipt weight; owner workers drive the environment from their
current-self/cached-teammate joint policy profile. Fixed-universe reset,
topology-motion, and the topology--refresh--receipt event decomposition are
exactly tested. The remaining bridge is statistical and performance-facing:
neural alignment calibration and equal-resource oracle headroom.

## Pursuit estimator closure (2026-09-08)

The delayed launch--receipt packet interface and the exact compatible
pair-factor candidate algebra pass.  In particular, exhaustive neighbor-action
enumeration, a centered directional finite difference, fixed actor reverse
calls, selected-packet taint exclusion, and exact copied-state branch replay
all agree.

The statistical bridge does not.  A privileged 875-dimensional bilinear head
failed on untouched conditional-mean data (`R^2=-1.0684`).  A deployable
summary-feature profile critic trained only from 1,024 selected completed
packets failed three of four gates.  The final raw-observation shared-CNN
factor critic also failed all four frozen development gates: edge-effect
`R^2=-0.0263`, nonzero sign accuracy `0.5676`, best-with-null accuracy `0.50`,
and edge/null alignment scale `0.0190`.

Pursuit is therefore stopped as the principal efficacy benchmark.  It remains
a structural stress test for dynamic causal support, recipient caches, exact
policy-byte charging, and topology-motion accounting.  No further Pursuit
estimator, calibration, pilot, formal run, or GPU job is authorized.  This
does not alter the finite-state positive core result or the estimator-agnostic
paired Lyapunov theorem; it leaves the standard-task efficacy bridge open.

Any replacement standard benchmark must be selected before outcome inspection
using four properties: local policy dependency, sufficiently informative
shaped learning signal, nontrivial cache-refresh oracle value at matched
transitions and bytes, and a parameter-shared critic whose candidate scan is
linear in declared local degree.  It must receive a new benchmark contract and
independent seeds.  Pursuit data cannot tune that contract.

The outcome-free selection now names Multiwalker as the primary contract
candidate because its per-step package-progress reward and public neighbor
observations directly address the signal/locality failure exposed by Pursuit.
Its isolated Box2D runtime, five-recipient policy caches, random delayed
receipt ledger, exact byte charging, and deterministic replay now pass.  The
continuous-action factor algebra also passes a double-precision directional
finite-difference oracle: four actor reverse calls are independent of degree
and the critic work is `1+2 Delta`.  These are structural results, not positive
learning evidence.  KAZ is the secondary heterogeneous-role stress task.  Both
still require a matched-resource oracle headroom gate before any efficacy
pilot.

The first Multiwalker development headroom matrix is retained as a `6/7`
failure. Its aggregate and effect gates pass, but the active directional rate
is `11/16=68.75%` rather than the frozen `75%`. The failure also exposes that
the labelled eight-step oracle is only receding-horizon greedy: under a prefix
budget it can be beaten by feasible age, mismatch, or random schedules and is
not an upper bound. No critic is authorized. A single Amendment may replace
that oracle by an exact full-horizon budget-aware optimizer while preserving
the states, costs, comparator family, seeds, and thresholds; failure of the
corrected kill test stops Multiwalker.

The prospectively frozen Amendment now passes all eight exact-oracle gates on
the same immutable development data. The zero-gap prefix-budget optimizer
beats the strong online envelope in `16/16` active cells, recovers `59.18%` of
the dynamic gain left above that envelope, and has `1.6119%` median normalized
headroom. This repairs the diagnostic but does not convert reused development
seeds into confirmation or establish an observable controller. The only
authorized next step is an untouched-seed repetition of both the comparator
envelope and exact oracle under an independently frozen protocol.

That independent repetition now passes C1--C9 on untouched seeds
`96100--96107`, with byte-exact clean reproduction. The exact optimizer beats
the strong online envelope in all `16/16` active cells, recovers `54.53%` of
the active oracle gain left above the envelope, and has `1.2870%` median
normalized headroom. Multiwalker therefore passes the problem-value gate:
there is a reproducible dynamic policy-cache allocation opportunity under the
declared communication constraint. This still does not validate an observable
controller, because the exact oracle sees future branch returns and evaluates
against a public all-current prefix. The next bridge is a frozen
delayed-feedback confidence interface and then causal cached-profile rollouts.

An immediate theory--experiment audit narrows that bridge: the confirmed
oracle optimizes finite-horizon cached-profile return, while the core
convergence theorem requires conditional owner-gradient packet alignment.
Return headroom is necessary evidence that refreshes change useful behavior,
but it does not imply alignment headroom. The next CPU gate must therefore
construct a privileged, fully charged conditional alignment oracle and compare
it with the same strong scheduling family before any selected-only critic is
fit. The launch--receipt ledger and exact reference-error/smoothness
decomposition are now executable; Multiwalker alignment headroom itself
remains open.

MW-AH-DEV-001 now closes that question negatively. The frozen sixteen-cell
CPU gate is byte-exactly reproduced and passes H1--H10 and H12, but fails H11:
the active median headroom is only `1.5224e-06` of ideal reference descent,
versus the preregistered `0.005` threshold. The high `94.60%` recovery ratio
only recovers most of an extremely small cache-sensitive gradient effect.
Consequently, no selected-feedback alignment critic, confirmation run, or GPU
experiment is authorized for this formulation.

The independent cached-profile return result remains true and estimates a
different object. Any continuation must explicitly reformulate the launch
objective around causal cache-state utility while using only the realized
selected packet to choose its receipt weight. It may not equate return with
gradient alignment or reuse the privileged MW-AH branches for training.
