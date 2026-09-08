## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: post-outcome theory and feasibility redesign
- Origin Date: 2026-09-09
- Verification Status: EXACT DECOMPOSITION; LEARNING AND MARL THEOREMS OPEN
- Version Label: drift_structured_relative_value_program_v1

# Drift-structured relative value for causal policy freshness

## Why the potential-only bridge stops

MW-PF-DEV-001 tests the low-complexity rule obtained by treating negative
cache and queue potential as the whole continuation value.  Its frozen primary
analysis selects zero cache-reset weight in every leave-one-seed-out fold and
does not beat the strengthened online envelope.  This is a failure of the
potential-only continuation model, not evidence that the previously measured
dynamic cache oracle is invalid.  The frozen experiment must be reproduced and
reported before this document can be used to authorize another experiment.

No alternative cache norm, reset coefficient, threshold, or seed is fitted in
this note.  The only admissible redesign is to represent the missing
intertemporal value explicitly.

## One controlled process

Index every centralized-training event by `t`.  The augmented state `X_t`
contains the Markov-game state, distinct actor parameters, recipient-specific
policy caches, the in-flight packet ledger, the communication queue, and the
event type.  At a launch event, the action is a null or one-edge cache refresh.
At a receipt event, the action is a scalar packet weight.  The two actions are
causally ordered and are never described as one simultaneous QP.

Let `r_t(X_t,a_t)` be the declared team utility at a launch and zero at a
receipt.  Let

\[
 P_t(X)=W F(\theta)+H_t(\theta,\chi)+\frac{Q^2}{2\nu}
\tag{1}
\]

be the paired learning/cache/resource Lyapunov potential.  The topology-robust
core sets `H=0`.  A cache extension uses one fixed edge universe.  If its
weights change, its transition contains the exact topology-motion term

\[
 \Xi_t=\frac12\sum_{e\in U}(\beta_{e,t+1}-\beta_{e,t})
 \|\theta_{j(e),t}-\chi_{e,t}\|^2.
\tag{2}
\]

Thus high topology turnover is never free: it either does not enter the core
potential or appears through `Xi_t`.

If every policy--cache distance is at most `R_chi`, its positive part has the
deterministic bound

\[
 \sum_{t<T}\Xi_t^+
 \le \frac{R_\chi^2}{2}
 \sum_{t<T}\sum_{e\in U}[\beta_{e,t+1}-\beta_{e,t}]_+.
\tag{2a}
\]

For binary weights `beta_(e,t)=beta 1{e active at t}`, the last sum is `beta`
times the number of edge activations.  A high-turnover Pursuit graph therefore
has an explicit topology-variation price.  It can be a favorable empirical
regime only when its learning/utility gain exceeds this price; turnover cannot
be advertised as useful while being omitted from the proof.

## Exact value decomposition

For a finite event horizon, let

\[
 V_t^\star(x)=\max_\pi\mathbb E_\pi
 \left[\sum_{s=t}^{T-1}r_s(X_s,A_s)\mid X_t=x\right]
\tag{3}
\]

over the declared feasible scheduler class.  Define the drift-structured
relative value

\[
 h_t^\star(x)=V_t^\star(x)+P_t(x)/V.
\tag{4}
\]

Substitution into the Bellman equation gives the identity

\[
 h_t^\star(x)=\frac{P_t(x)}{V}+
 \max_a\left\{r_t(x,a)+
 \mathbb E\left[-\frac{P_{t+1}(X')}{V}
                    +h_{t+1}^\star(X')\mid x,a\right]\right\}.
\tag{5}
\]

Consequently the exact action score, up to an action-independent constant, is

\[
 r_t(x,a)-\frac1V\mathbb E[\Delta P_t\mid x,a]
 +\mathbb E[h_{t+1}^\star(X')\mid x,a].
\tag{6}
\]

Classical one-step drift-plus-penalty is exactly the special case that drops
the last term.  MW-PF-DEV-001 is an empirical test of that special case for the
cache launch action.  Equation (6), not a new mismatch heuristic, is the only
principled way forward after its failure.

## Executable two-stage controller

At launch, a bounded-degree factor model estimates the residual successor
value for the null action and each eligible directed edge.  The scheduler
chooses

\[
 a_t\in\arg\max_{a\in\mathcal A_t}
 \left\{\widehat U_t(a)-\widehat{\Delta P_t}(a)/V
       +\widehat{\mathbb E h_{t+1}}(a)+b_t(a)\right\},
\tag{7}
\]

subject to the hard prefix byte cap.  `b_t` is a simultaneous confidence
radius, not an uncharged probe.  Only the selected cached profile is executed
and only its fully charged trajectory is observed.  Candidate computation is
`O(Delta d)` for a diagonal linear residual model, where `Delta` is the local
policy-dependency degree and `d` the feature dimension.

At receipt, the returned packet and current actor/cache state determine the
certified scalar smoothness upper bound.  Its packet weight is the clipped
one-dimensional minimizer already derived in the paired-freshness note.  The
receipt action is not predicted using a counterfactual packet at launch.

## Performance implication of a learned residual

Let `hhat_(t+1)` approximate (4) up to the shift-invariant error

\[
 \epsilon_t=\inf_{c\in\mathbb R}\sup_x
 |\widehat h_{t+1}(x)+c-h_{t+1}^\star(x)|,
\tag{8}
\]

and let all implemented score errors, including queue linearization and
topology motion, be at most `zeta_t`.  Backward induction then gives

\[
 V_0^\star(x)-V_0^\pi(x)
 \le 2\sum_{t=0}^{T-1}(\epsilon_t+\zeta_t).
\tag{9}
\]

This is an induced-trajectory statement.  It is stronger than contextual
action regret, but it is useful only after an explicit learning bound controls
(8).  The decomposition itself does not provide that bound for free.

## The theorem that would make the paper viable

A complete positive theorem must use the same event process as the code and
establish all of the following:

1. delayed selected-feedback learning of a realizable or approximately
   realizable residual value, with a finite cumulative confidence term;
2. dynamic scheduler regret from (6), not merely action regret on the states
   visited by the proposed controller;
3. chronological launch--receipt telescoping and a terminal outstanding-packet
   boundary or an explicit drain protocol;
4. owner-block stationarity under Markov trajectory bias and delayed receipt;
5. pathwise communication feasibility from the same virtual queue;
6. an explicit `sum Xi_t^+` term for every moving-weight cache theorem;
7. complexity proportional to sparse local degree, with dense games reported
   as an honest degeneration.

If these conditions hold under linear residual realizability, the intended
rate has the qualitative form

\[
 \mathrm{Reg}_T^{\rm sched}
 +W\sum_{p<T}\kappa_p
   \mathbb E\|\nabla_{i_p}F(\theta_{b(p)})\|^2
 \lesssim
 \widetilde O\!\left(d\sqrt{T(D+1)}\right)
 +\mathrm{MarkovBias}+\mathrm{ApproxErr}
 +\sum_t\mathbb E\Xi_t^+ +P_0.
\tag{10}
\]

Equation (10) is a target, not a proved rate.  In particular, a linear-MDP or
Bellman-completeness assumption must be stated rather than hidden inside a
generic neural approximation term.

## Novelty boundary and stop rule

Combining long-horizon RL with drift-plus-penalty is not by itself novel.
Neither is learning communication graphs, delayed policy-gradient clipping,
or Lyapunov analysis.  The paper is viable only if the complete contribution
is specific to asynchronous CTDE with distinct interacting actors:

- a launch refresh changes the joint behavior kernel that generates one
  owner's packet;
- a delayed receipt changes only that owner's actor block;
- one event-state value decomposition couples online team utility, learning
  stationarity, and actual policy-byte feasibility;
- a sparse policy-dependency factorization makes the controller executable;
- standard MARL experiments show a Pareto improvement over strong static,
  age-based, DPP-only, scheduler-RL-only, and oracle-informed baselines.

Before a new experiment identifier is preregistered, two stop gates are
mandatory.  First, a proof audit must close the exact event timing and specify
a residual function class whose learning guarantee is valid.  Second, an
outcome-labelled CPU feasibility study using actual cache-dependent rollouts
must show material headroom over `oracle_h8` and the other strong online
baselines.  Failure of either gate stops this redesign; thresholds or seeds may
not be changed after observing the result.
