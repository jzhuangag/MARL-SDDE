## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: post-outcome formulation and theorem-feasibility audit
- Origin Date: 2026-09-08
- Verification Status: CANDIDATE FORMULATION; NOT A PREREGISTRATION OR RESULT
- Version Label: paired_freshness_reformulation_v1

# Act Fresh, Update Safely: paired Lyapunov control for asynchronous MARL

## Decision forced by the two Multiwalker diagnostics

The independently confirmed Multiwalker cache oracle shows material dynamic
`H`-return value under a prefix policy-byte budget.  MW-AH-DEV-001, by
contrast, shows that the same cache decision has negligible headroom in the
one-update owner-gradient alignment objective.  These results are compatible:
a teammate cache can change the joint trajectory and its return without
materially rotating the derivative of that return with respect to the owner's
first action.

The next candidate must therefore not identify cached-profile return with
one-step gradient alignment.  The aligned-controller line is stopped.  The
remaining coherent question is:

> During asynchronous CTDE, can one causal Lyapunov policy decide which
> teammate policy version a worker should act with, and later how strongly to
> apply the resulting stale packet, so as to maximize online team performance
> under a policy-byte budget while retaining a finite-time learning guarantee?

This is one packet-lifecycle problem, not two unrelated modules.  The launch
decision changes the behavior kernel; the receipt decision changes the
parameter iterate.  Execution remains decentralized: the final local actors
do not exchange policy caches.

## Event model and single objective

At launch `p`, owner worker `i_p` has recipient cache `chi_p`.  It chooses

\[
 a_p\in\{\varnothing\}\cup E_p(i_p),
\]

where one non-null action refreshes one directed teammate cache and costs
`c_p(a_p)` policy bytes.  The resulting cached joint policy induces a
controlled Markov kernel and a delayed trajectory packet.  Let

\[
 U_p(a)=\mathbb E[R_p^H\mid\mathcal F_{b(p)},a]
\]

be its finite-horizon team utility.  When the selected packet returns, the
learner applies its owner gradient `G_p(a_p)` with an online weight
`alpha_p in [0,alpha_max]`.

The target is cumulative team utility subject to an average communication
constraint, together with a finite-time stationarity guarantee for the
all-current cooperative objective `F`:

\[
 \max_\Pi\ \mathbb E\sum_{p<N}U_p(a_p),\qquad
 \limsup_N\frac1N\mathbb E\sum_{p<N}c_p(a_p)\le\bar c,
\tag{1}
\]

while bounding a weighted sum of owner-block gradients.  Wall-clock time is a
secondary systems metric.  Delay remains fundamental because it separates
launch feedback from receipt and increases parameter motion.

Use the communication queue

\[
 Q_{p+1}=[Q_p+\nu(c_p(a_p)-\bar c)]^+
\]

and the topology-robust Lyapunov function

\[
 \mathcal L_t=W F(\theta_t)+\frac{Q_t^2}{2\nu}.
\tag{2}
\]

The cache-energy extension is not part of (2).  On a changing topology it is
valid only on a fixed edge universe and with the explicit positive
topology-motion remainder `Xi_p^+`.

## Paired launch--receipt drift

Let `v_p` be a launch-measurable reference direction formed only from packets
completed before launch.  Suppose `||nabla_i F(theta_b)-v_p||<=e_p`,
`||G_p||<=G_p^max`, block smoothness is `L_i`, and launch-to-receipt owner
motion is at most `M_p`.  At receipt, the selected packet gives the observable
scalar `Y_p=<v_p,G_p>`.  The paired objective drift obeys

\[
 \Delta F_p\le
 -\alpha_p\{Y_p-G_p^{\max}(e_p+L_iM_p)\}
 +\frac{L_i(G_p^{\max})^2}{2}\alpha_p^2.
\tag{3}
\]

Unlike the failed alignment formulation, (3) is not used to predict which
cache edge is valuable.  It decides how much to trust the packet that the
chosen cached behavior actually produced.  Its receipt-time minimizer is the
one-dimensional certified root

\[
 \alpha_p^*=\Pi_{[0,\alpha_{\max}]}
 \frac{[Y_p-G_p^{\max}(e_p+L_iM_p)]_+}
      {L_i(G_p^{\max})^2}.
\tag{4}
\]

Thus `alpha` is online and Lyapunov-designed, but no counterfactual gradient
alignment model is required.

At launch, an optimistic utility model `U_p^U(a)` based only on previously
returned selected trajectories gives

\[
 a_p\in\arg\min_{a\in\mathcal A_p^{\rm feasible}}
 \{-V U_p^U(a)+Q_pc_p(a)\}.
\tag{5}
\]

The feasible set enforces the hard prefix cap.  Equation (5) is not merely a
myopic reward rule if the utility state includes the persistent recipient
cache: the learned object is the value of a cache-state transition, not the
immediate reward of transmitting a message.  This is precisely the dynamic
value isolated by the exact prefix-budget oracle.  The intended low-complexity
realization uses a bounded-degree factorized critic and null-plus-one-edge
actions, so candidate evaluation is `O(Delta)` after local features are
formed.

Pairing the launch queue increment, the selected utility penalty, and the
eventual receipt drift yields

\[
 \Delta\mathcal L_p-VU_p(a_p)
 \le Q_p(c_p(a_p)-\bar c)-VU_p(a_p)
 +W\,D_p^{\rm rec}(\alpha_p)+B_Q,
\tag{6}
\]

where `D_p^rec` is the right side of (3) and
`B_Q=nu(c_p-bar c)^2/2`.  This is the common proof object for (4) and (5).
They occur at different causal times, so describing them as a simultaneous
QP would be incorrect.

### Conditional paired theorem

Suppose the selected-feedback utility model is optimistic on one simultaneous
event,

\[
 0\le U_p^U(a)-U_p(a)\le 2r_p(a),
\]

and let `a_p^o` be any launch-measurable comparator whose conditional expected
cost is at most `bar c`.  Exact minimization of (5) gives

\[
 [-VU_p(a_p)+Q_pc_p(a_p)]-
 [-VU_p(a_p^o)+Q_pc_p(a_p^o)]
 \le 2V r_p(a_p).
\tag{6a}
\]

If the receipt rule additionally satisfies the estimator-specific condition

\[
 \mathbb E[D_p^{\rm rec}(\alpha_p)\mid\mathcal F_{b(p)}]
 \le-\kappa_p\|\nabla_{i_p}F(\theta_{b(p)})\|^2+R_p,
\tag{6b}
\]

then chronological launch--receipt pairing and the queue half-square inequality
yield

\[
\begin{aligned}
 V\sum_{p<N}\mathbb E[U_p(a_p^o)-U_p(a_p)]
 &+W\sum_{p<N}\kappa_p
   \mathbb E\|\nabla_{i_p}F(\theta_{b(p)})\|^2\\
 \le{}&W(F(\theta_0)-F_\star)+\frac{Q_0^2}{2\nu}
 +2V\sum_{p<N}\mathbb E r_p(a_p)\\
 &+NB_Q+W\sum_{p<N}\mathbb E R_p .
\end{aligned}
\tag{6c}
\]

For the optional variable-weight cache-energy extension, the right side also
contains `sum_p E[Xi_p^+]`; the topology-robust core bound does not.  The proof
of (6a) is the standard optimistic sandwich, but only the selected action's
radius is paid.  For (6c), add the receipt inequality and queue drift to (6a),
use the comparator's conditional cost feasibility, telescope (2), and use
`F>=F_star`.  The remaining hard work is not this algebra: it is verifying the
controlled-Markov confidence and receipt condition (6b) for an executable
critic.

## What a complete theorem must prove

The finite-event queue and receipt algebra in (2)--(6) is exact.  An ICML-level
theorem still requires all of the following, without hiding them in a neural
approximation term:

1. a controlled-Markov utility model with selected delayed feedback and an
   explicit regret term relative to a feasible cache-state policy;
2. a reference-direction and launch-to-receipt error bound giving (3)
   simultaneously over time;
3. an epoch conversion from owner-block descent to full-gradient stationarity,
   including within-epoch receipt-path motion;
4. an explicit cache-induced Markov-gradient bias term, or a mandatory refresh
   skeleton that makes this bias vanish with the learning rate;
5. queue stability and the pathwise prefix-budget relation;
6. the exact `sum Xi_p^+/W` remainder for any optional moving-topology cache
   energy.

Only after these obligations close may (6c) be converted to the full-gradient
and explicit-rate bound of the form

\[
 \text{utility regret}+W\sum_p\kappa_p
 \mathbb E\|\nabla F(\theta_{s(p)})\|^2
 \le \text{initial potential}+\text{delay}+
 \text{Markov bias}+\text{estimation}+
 \text{topology motion}.
\tag{7}
\]

Equation (7) is a theorem target, not a proved result in this note.

## Why this can still be one ICML story

The scientific object is policy-version freshness inside asynchronous MARL.
It has two causal consequences that existing shared-parameter asynchronous
optimization does not contain: the version selected at launch changes which
joint policy acts in the Markov game, and its delayed packet later changes one
distinct actor block.  The same paired Lyapunov argument prices the former by
team utility and the latter by certified receipt drift, under one byte queue.

The novelty cannot be claimed from Lyapunov optimization, delayed bandits,
dynamic graphs, CTDE, or adaptive packet weights individually.  It depends on
the complete training-time policy-cache lifecycle, sparse local factorization,
matched policy-byte accounting, and simultaneous online-performance and
learning guarantees.  The current bounded literature comparison is recorded
in `novelty_confrontation_policy_cache_graph_20260908.md`; a fresh systematic
audit remains mandatory before submission.

## Non-negotiable empirical gates

The next experiment is not authorized by this design note.  Before any GPU
training, an independently frozen CPU gate must establish all of the following:

1. selected-feedback cache-state value can be learned without consuming the
   privileged counterfactual or confirmation-oracle data;
2. the causal controller recovers a material fraction of the already confirmed
   return headroom against the full strong online envelope;
3. actual cached rollouts, rather than an all-current public prefix, drive the
   state evolution;
4. every diagnostic and learning transition and every transmitted policy byte
   is charged;
5. the receipt root (4) is nontrivial and improves stability under injected
   service delay relative to fixed, age-scaled, and trust-region packet weights;
6. graph-only, weight-only, queue-free, and full paired-controller ablations
   separate the two causal effects;
7. dense/topology-turnover cases degrade honestly and report `Xi_p^+` where
   cache energy is used.

Passing such a CPU gate would justify a new preregistered standard CTDE pilot.
Failure would stop this reformulation.  It would not be repaired by changing
the already observed MW-AH-DEV-001 threshold or training on its privileged
counterfactuals.

## Role of an SDDE

The exact event-time recursion is primary.  A hybrid controlled SDDE may be
derived only after a finite-horizon coupling result: continuous parameter
motion between events, jump resets of recipient caches, delayed packet marks,
and reflected queue dynamics.  Itô--Lyapunov analysis could then expose a
delay--budget--interaction phase boundary and guide scaling of `V`, `W`, and
`nu`.  Without that coupling, an SDDE remains an interpretation and must not
replace the discrete convergence or regret proof.
