## Material Passport

- Origin Skills: academic-research-suite; publishable-academic-writing
- Origin Mode: contribution-led paper architecture
- Origin Date: 2026-09-07
- Verification Status: WORKING BLUEPRINT; STANDARD-MARL EFFICACY AND NEURAL RADIUS PENDING
- Version Label: causal_policy_freshness_paper_blueprint_v1

# Causal Policy Freshness: ICML 2027 paper blueprint

## One paper, one question

**How should an asynchronous CTDE learner spend a limited policy-synchronization
budget when a rollout worker's stale teammate policies can either help or harm
the next owner update?**

The answer is not “refresh the oldest policy,” “minimize parameter mismatch,”
or “downweight every stale gradient.”  In a Markov game, a cache refresh changes
the joint behavior policy and hence the trajectory distribution of a delayed
future packet.  Its value is signed, state dependent, and learned only after
the packet returns.  The proposed method treats this as one causal stochastic
control problem.

## The single causal chain

1. Distinct agents have distinct actor parameters.  An owner worker launches a
   trajectory with its current actor and cached teammate actors.
2. A directed cache edge `j -> i` changes only the behavior profile used to
   generate agent `i`'s next packet; it does not average actor parameters.
3. A locally factored critic assigns a signed finite-horizon alignment feature
   to the null action and each eligible one-edge refresh.
4. The alignment feedback arrives with the trajectory.  A delayed optimistic
   local linear model learns which edge has useful drift value.
5. One composite Lyapunov upper bound jointly chooses the edge and the packet's
   receipt-time weight while a virtual queue prices actual policy bytes.
6. The exact launch/receipt recursion yields a finite-time stationarity bound
   relative to a dynamic feasible comparator and a pathwise average-byte bound.

Every theorem and experiment must instantiate one link in this chain.  A
result that concerns only generic participation, generic delayed SGD, an
execution-time communication graph, or a shared global policy does not support
the paper's claim.

## Algorithmic object

At launch event `p`, owner `i_p` has local candidate set

\[
 \mathcal A_p=\{\varnothing\}\cup
 \{j\to i_p:j\in\mathcal N_p(i_p)\},
 \qquad |\mathcal N_p(i_p)|\leq\Delta.
\]

For each candidate the delayed local factor model supplies an optimistic
alignment `A_p^U(a)`.  The controller exactly minimizes

\[
 J_p(a,\alpha)
 =-m_p(a;A_p^U)\alpha+\frac{C_p(a)}2\alpha^2
  -B_p(a)+Q_pc_p(a),
\]

using

\[
 \alpha_p(a)=\Pi_{[0,\bar\alpha]}
 \left(\frac{m_p(a;A_p^U)}{C_p(a)}\right)
\]

and an `O(Delta)` candidate scan.  `B_p(a)` is the exact strategic-cache
energy reset and `Q_p c_p(a)` is the actual communication price.  Delay,
launch-to-receipt motion, critic uncertainty, and action-specific gradient
bounds enter `m_p` or the proved estimation remainder; none is represented by
an arbitrary age-decay coefficient.

The online variables are therefore the directed training-time collaboration
edge and packet weight.  The final actors execute independently with local
observations; the graph is not an execution protocol.

## Main theorem chain

The paper should present one theorem with four explicit lemmas, not unrelated
claims.

### Lemma A: exact paired Lyapunov drift

Chronologically telescope launch, cache replacement, delayed receipt, owner
update, outgoing-cache change, and graph-support change for

\[
 \mathcal L_p=V F(\theta_p)+H_p+Q_p^2/(2\nu).
\]

This establishes the candidate-specific quadratic index and its scalar closed
form.  The graph-switch positive increment remains explicit.

### Lemma B: controlled Markov-game kernel

A one-edge policy refresh changes the joint behavior policy by at most its
local categorical policy TV.  Coupling bounds the `H`-step trajectory-score
shift by the sum of candidate-specific marginal TV terms.  This prevents the
tabular common-kernel simplification from being silently transferred to
Pursuit.

### Lemma C: delayed optimistic drift regret

For a predictable local linear alignment model, self-normalized confidence and
the optimistic Lyapunov minimizer give

\[
 J_p(u_p;A_p)-J_p(u_p^\circ;A_p)
 \leq 2V\bar\alpha r_p(a_p).
\]

Only the selected edge's radius appears.  If feedback returns within `D`
launches, a residue-class elliptical-potential argument bounds cumulative
leverage by order

\[
 \sqrt{N(D+1)d\log(1+NL_x^2/(\lambda d))}.
\]

The bandit concentration machinery is inherited; its coupling to the true
MARL drift decision is the paper-specific step.

### Main theorem: learning and communication

Insert Lemma C into Lemma A and compare with a conditionally budget-feasible
dynamic policy.  The result bounds average owner-block stationarity by initial
potential/cache energy, graph-switch and Markov/factor approximation, stochastic
packet variance, delayed alignment learning, and the queue tradeoff.  The queue
iteration simultaneously gives the pathwise average policy-byte constraint.

The theorem should not claim last-iterate Nash convergence for an unrestricted
general-sum game.  The primary scope is cooperative potential Markov games and
owner-block stationarity under explicitly stated approximation conditions.

## Why Lyapunov is essential rather than decorative

The learning potential `F` says whether applying a packet is useful.  Cache
energy `H` records which teammate versions make that packet strategically
stale.  Communication debt `Q` records whether repeated refreshes are feasible.
Minimizing the drift of their sum produces both controlled variables.  Removing
any one term changes the executable action:

- without `F`, the rule refreshes large mismatches even when staleness is
  beneficial;
- without `H`, uncertain edge learning can make the null graph absorbing;
- without `Q`, the scheduler has no enforceable long-run byte budget;
- without the receipt-weight minimization, a correctly refreshed but very late
  packet can still destabilize the owner update.

This four-way ablation is the empirical counterpart of the theorem, not a list
of unrelated engineering options.

## Role of the SDDE

The exact event-time recursion is the main model because cache refreshes are
jumps and launch/receipt events are discrete.  A controlled hybrid SDDE is
useful only as an appendix corollary if a small-step generator or finite-horizon
weak-coupling result is proved.  It can then visualize how service delay,
interaction strength, and queue price move the favorable phase boundary.  It
must not replace the exact discrete convergence theorem or appear solely to
make the paper sound more mathematical.

## Evidence package

### Mechanism and theorem evidence (CPU)

- exact algebra and exhaustive finite-state tests for every drift identity;
- an analytic forecast-reversal phase where the best instantaneous edge differs
  from the best finite-horizon edge;
- independent untouched-seed confirmation of the complete plug-in controller,
  including strong online myopic, fixed, periodic, random, and exact-oracle
  comparators;
- delayed-UCB confidence, selected-action regret, queue, and controlled-kernel
  tests;
- dense-graph degeneration and degree/runtime scaling.

The tabular mechanism can appear as a theorem illustration or appendix result.
It cannot substitute for a standard MARL benchmark.

### Standard MARL evidence (Pursuit; GPU only after interface qualification)

The positive benchmark uses distinct actor blocks, asynchronous owner launches,
random service delay, state-dependent degree-four local factors, a centralized
training critic, and decentralized evaluation.  Primary outcomes are return
and sample efficiency versus actual policy bytes; wall-clock and estimator
overhead are secondary.

The strong comparator family includes no optional refresh, complete refresh,
age, parameter mismatch, fixed local graphs/rates, periodic refresh, a greedy
myopic signed scheduler, and their resource-feasible envelope.  Primary claims
require broad improvement over the strongest non-oracle online baseline, not
only over one weak fixed graph.  Ablations remove optimism, signed alignment,
cache energy, packet-weight control, state-dependent graph support, and queue
pricing one at a time.

Pistonball remains the dense degeneration case already falsified for the
low-degree claim.  It is not rerun until a new outcome-free reason exists.

## Paper organization

1. Introduction: the mixed-policy rollout created by asynchronous CTDE.
2. Problem: directed policy caches, event ledger, resource constraint, and
   cooperative potential objective.
3. Method: local factor alignment model and joint optimistic Lyapunov action.
4. Theory: the four-link theorem chain above.
5. Experiments: tabular phase/confirmation, Pursuit learning frontier,
   ablations, scaling, and dense degeneration.
6. Related work: shared-policy actor--learner lag, homogeneous cooperative
   linear MDPs, delayed bandits, coordination graphs, constrained RL, and SDDE
   distributed SGD.
7. Limitations: factor approximation, non-informative service delay, local
   potential-game scope, and absence of universal no-harm.

## Remaining kill gates

The project is not ICML-ready until all of the following are true:

1. the independent forecast-reversal confirmation and clean reproduction pass
   every frozen gate;
2. the delayed optimistic alignment implementation matches its theorem and has
   a nonvacuous CPU calibration interface;
3. an outcome-free Pursuit oracle-value audit shows material equal-resource
   headroom over the full strong online family;
4. a preregistered Pursuit pilot yields broad return--communication gains with
   acceptable overhead;
5. independent seeds reproduce the principal benchmark direction;
6. the main theorem carries all critic, factor, controlled-kernel, graph-switch,
   and delay terms without double counting;
7. a fresh systematic novelty and citation-integrity audit finds no directly
   subsuming method.

Failure of a gate changes the claim or stops the standard-benchmark line; it is
not repaired by renaming the experiment or weakening a frozen comparator.
