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
5. One topology-robust Lyapunov upper bound jointly chooses the edge and the
   packet's receipt-time weight while a virtual queue prices actual policy
   bytes. A fixed-universe cache potential is an optional strengthening, not a
   hidden assumption of the main guarantee.
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
alignment `A_p^U(a)`. The principal controller exactly minimizes

\[
 J_p^{\rm core}(a,\alpha)
 =-V[A_p^U(a)-L_iG_p(a)M_p]\alpha
  +\frac{VL_iG_p(a)^2}{2}\alpha^2+Q_pc_p(a),
\]

using

\[
 \alpha_p(a)=\Pi_{[0,\bar\alpha]}
 \left(\frac{m_p^{\rm core}(a;A_p^U)}
 {C_p^{\rm core}(a)}\right)
\]

and an `O(Delta)` candidate scan. `Q_p c_p(a)` is the actual communication
price. Delay, launch-to-receipt motion, critic uncertainty, and action-specific
gradient bounds enter the same bound; none is represented by an arbitrary
age-decay coefficient. The edge remains an optimization variable because
refreshing it changes the action-specific future trajectory alignment.

For persistent caches, a secondary controller augments the index with an exact
cache reset and outgoing-cache motion. Its cache energy is defined on one fixed
edge universe. If relevance weights change, the theorem carries the exact
topology-motion remainder. It never sums cache energy only over the currently
active edges and then assumes that sum telescopes.

The online variables are therefore the directed training-time collaboration
edge and packet weight.  The final actors execute independently with local
observations; the graph is not an execution protocol.

## Main theorem chain

The paper should present one theorem with four explicit lemmas, not unrelated
claims.

### Lemma A: topology-robust paired Lyapunov drift

Chronologically telescope launch, delayed receipt, owner update, and the
communication queue for

\[
 \mathcal L_p^{\rm core}=V F(\theta_p)+Q_p^2/(2\nu).
\]

This establishes the candidate-specific quadratic index and its scalar closed
form without a topology-motion assumption. A cache-extension lemma then adds
`H` on a fixed edge universe. If its weights change, the exact remainder is

\[
 \Xi_p=\frac12\sum_{e\in U}(\beta_{e,p+1}-\beta_{e,p})
 \|\theta_{j(e),p}-\chi_{e,p}\|^2.
\]

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

Insert Lemma C into the core Lemma A and compare with a conditionally
budget-feasible dynamic policy. The result bounds average owner-block
stationarity by initial potential, Markov/factor approximation, stochastic
packet variance, delayed alignment learning, and the queue tradeoff. It is
valid under arbitrary state-dependent candidate-set turnover. The optional
cache theorem adds initial cache energy and the positive topology-motion sum.
The queue iteration simultaneously gives the pathwise average policy-byte
constraint.

The theorem should not claim last-iterate Nash convergence for an unrestricted
general-sum game.  The primary scope is cooperative potential Markov games and
owner-block stationarity under explicitly stated approximation conditions.

## Why Lyapunov is essential rather than decorative

The learning potential `F` says whether the trajectory produced by an edge and
its delayed packet are useful. Communication debt `Q` says whether repeatedly
choosing such edges is feasible. Minimizing the drift of `VF+Q^2/(2nu)`
produces both controlled variables: the edge through signed trajectory
alignment and the receipt weight through a scalar quadratic minimizer. This is
the main design, not a post-hoc proof.

Cache energy `H` is a principled optional memory term: it records persistent
policy-version mismatch and can create extra freshness pressure. Its scope is
deliberately narrower because topology motion must be paid. Removing terms has
the following causal effects:

- without `F`, the rule refreshes large mismatches even when staleness is
  beneficial;
- without `Q`, the scheduler has no enforceable long-run byte budget;
- without the receipt-weight minimization, a correctly refreshed but very late
  packet can still destabilize the owner update;
- adding fixed-universe `H` creates persistent cache pressure, whereas
  active-edge-only `H` creates an unaccounted jump whenever the graph turns
  over.

The core-versus-cache ablation is therefore a theorem-scope test, not a list of
unrelated engineering options.

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
- independent untouched-seed confirmation of the core plug-in controller,
  including strong online myopic, fixed, periodic, random, and exact-oracle
  comparators;
- delayed-UCB confidence, selected-action regret, queue, controlled-kernel, and
  topology-motion identity tests;
- dense-graph degeneration and degree/runtime scaling.

The tabular mechanism can appear as a theorem illustration or appendix result.
It cannot substitute for a standard MARL benchmark.

### Standard MARL evidence (benchmark selection reopened after Pursuit stop)

The required positive benchmark must use distinct actor blocks, asynchronous
owner launches, random service delay, local policy factors, a centralized
training critic, and decentralized evaluation.  Primary outcomes are return
and sample efficiency versus actual policy bytes; wall-clock and estimator
overhead are secondary.  Pursuit no longer fills this role: its final
raw-observation factor critic failed all four frozen estimator-class gates on
2026-09-08.  Pursuit is retained only as a dynamic-topology/cache-semantics
stress test.

The strong comparator family includes no optional refresh, complete refresh,
age, parameter mismatch, fixed local graphs/rates, periodic refresh, a greedy
myopic signed scheduler, and their resource-feasible envelope.  Primary claims
require broad improvement over the strongest non-oracle online baseline, not
only over one weak fixed graph. Ablations remove optimism, signed alignment,
packet-weight control, state-dependent graph support, and queue pricing. The
cache-energy extension is reported separately with fixed-universe and
topology-motion accounting.

Pistonball remains the dense degeneration case already falsified for the
low-degree claim.  It is not rerun until a new outcome-free reason exists.  A
replacement positive benchmark is Multiwalker provisionally: its public
neighbor observations and per-step package-progress reward address locality
and signal density before outcome inspection.  Its outcome-free cache/receipt
contract and exact continuous-action factor algebra now pass, including a
degree-independent four-call actor Jacobian direction and an `O(Delta)` critic
candidate scan.  KAZ is the heterogeneous-role secondary task.  Neither
enters the paper as efficacy evidence until a separately frozen contract
establishes matched-resource oracle headroom and a deployable
parameter-shared factor critic.

## Paper organization

1. Introduction: the mixed-policy rollout created by asynchronous CTDE.
2. Problem: directed policy caches, event ledger, resource constraint, and
   cooperative potential objective.
3. Method: local factor alignment model and joint optimistic Lyapunov action.
4. Theory: the four-link theorem chain above.
5. Experiments: tabular phase/confirmation, a still-to-be-qualified standard
   MARL learning frontier, ablations, scaling, and Pursuit/Pistonball structural
   stress tests.
6. Related work: shared-policy actor--learner lag, homogeneous cooperative
   linear MDPs, delayed bandits, coordination graphs, constrained RL, and SDDE
   distributed SGD.
7. Limitations: factor approximation, non-informative service delay, local
   potential-game scope, and absence of universal no-harm.

## Evidence status and remaining kill gates

The independent forecast-reversal confirmation has passed every frozen gate
with byte-exact reproduction. Within the finite-state model, this closes the
core mechanism link: favorable risk ratio `0.587831`, all `12/12` favorable
cells and `384/384` paired seed-cells improve, and median exact-oracle headroom
recovery is `0.875652`. It does not close the persistent-cache extension's
performance or the standard-MARL links.

The outcome-free Pursuit state-machine qualification closes launch
cache copying, full byte charging, zero-weight persistence, owner-specific
cached behavior, and the exact topology-motion event decomposition with
byte-identical reproduction. Three increasingly structured estimator-class
audits then failed, including the final raw-observation shared CNN.  Pursuit is
therefore stopped as an efficacy benchmark and does not show that the
controller improves standard-task return.

The project is not ICML-ready until all of the following remaining conditions
are true:

1. the Multiwalker replacement contract and continuous factor algebra are
   extended by a nonvacuous statistical critic interface without using
   Pursuit outcomes;
2. an outcome-free oracle-value audit on that benchmark shows material
   equal-resource headroom over the full strong online family;
3. a preregistered standard-task pilot yields broad return--communication gains
   with acceptable overhead;
4. independent seeds reproduce the principal benchmark direction;
5. the main core theorem carries critic, factor, controlled-kernel, and delay
   terms without depending on graph-switch cancellation; any cache theorem
   separately carries its exact topology-motion term without double counting;
6. a fresh systematic novelty and citation-integrity audit finds no directly
   subsuming method.

Failure of a gate changes the claim or stops the standard-benchmark line; it is
not repaired by renaming the experiment or weakening a frozen comparator.

The development version of condition 2 is now positive after a prospectively
frozen correction from a greedy lookahead diagnostic to the exact 40-prefix
budget oracle. On the unchanged development seeds it yields `59.18%` recovery,
`16/16` active directions, and `1.6119%` median normalized headroom over the
strong online envelope. Condition 2 remains open at paper-evidence level until
the same calculation passes on untouched seeds. No learned critic or
controller result is implied by this oracle ceiling.

Condition 2 now also passes an independently frozen eight-seed confirmation:
all validity and performance gates pass, active direction is `16/16`, recovery
is `54.53%`, median normalized headroom is `1.2870%`, and the clean rerun is
byte-identical. The paper-level status nevertheless remains pre-controller.
The exact oracle consumes future simulator values and the qualification prefix
is a fixed all-current reference; a causal, learned score must next recover a
nontrivial share of this headroom on cache-dependent rollouts before standard
CTDE efficacy or GPU experiments are authorized.
