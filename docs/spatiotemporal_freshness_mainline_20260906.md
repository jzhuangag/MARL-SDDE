# Spatiotemporal freshness in asynchronous multi-agent policy learning

Date: 2026-09-06

Status: unified ICML-candidate research program.  The causal-cone certificate
has a theorem; the final controller and benchmark-return evidence are not yet
complete.

## Candidate title

**Spatiotemporal Freshness: Lyapunov Control of Rollout Horizons and Policy-
Version Graphs in Asynchronous Multi-Agent Reinforcement Learning**

## One-sentence contribution

We formulate asynchronous CTDE as a causal freshness-control problem and use
one composite Lyapunov function to choose how long each rollout should be,
which strategically relevant teammate-policy versions its worker should
refresh, and how much of the delayed packet should be applied when it returns.

## Why this is one story

A rollout horizon is simultaneously a statistical and a systems decision.
A longer trajectory can improve return/advantage estimation and expose slower
Markov modes.  It also expands the set of policies that can causally influence
the owner update, increases the policy-version payload, and gives concurrent
policies more time to change before the packet returns.  A shorter trajectory
has the opposite tradeoff.  The communication graph is therefore not an
independent architectural choice: it is the spatial footprint induced by the
temporal rollout decision.

The development Pistonball audit exhibits this phase.  Across three policy
profiles and three launch states, the median conformal-development tube cost
was approximately 23.7%, 34.7%, and 41.1% of the complete graph at horizons 4,
6, and 8.  Some center-state horizon-eight tubes were dense.  These values are
development measurements, not final evidence, but they rule out a fixed sparse
graph and motivate joint spatiotemporal control.

## Training and execution model

Training is centralized and asynchronous.  Each worker launches a joint
environment trajectory for one owner policy block using a local cache of all
teammate policy versions.  At launch, the server knows the current centralized
state, the worker's version vector, the communication queue, and the public
trajectory-cone certificate.  At receipt, it additionally knows the realized
packet gradient and all intervening policy versions.

Execution is decentralized.  Each agent uses its final local policy and local
observation.  No policy-cache graph, central server, or new execution message
is required.

## Decision variables

At dispatch epoch `k`, choose

\[
(H_k,E_k),\qquad E_k\subseteq C(X_k,H_k),
\tag{1}
\]

where `H_k` is the rollout horizon and `E_k` is the set of policy-version
refreshes inside the certified causal cone.  At receipt epoch `ell(k)`, choose
the applied mass `alpha_k`.  These variables occur at different filtrations and
must not be collapsed into one clairvoyant QP.

For every candidate horizon, nested cone shells supply a certified omission
remainder and an exact edge cost.  The launch decision minimizes

\[
V\widehat D_k(H,E)+Q_k c_k(H,E)
\tag{2}
\]

by selecting positive net-benefit shells/edges.  The receipt decision minimizes
the realized scalar quadratic drift bound and has the closed form already
proved for `alpha_k`.  Thus Lyapunov drift creates all three controls; it is not
added only after the algorithm is chosen.

## Theorem stack

1. **Causal-cone lemma:** exact graphical separation gives zero omitted
   finite-horizon policy effect; a coupled approximate cone gives an explicit
   `4 B_R B_Z delta_cone` gradient error.
2. **Shifted conformal certificate:** split conformal calibrates the trajectory
   tube at a reference policy; a trajectory-KL/Pinsker term corrects bounded
   policy drift.
3. **Noisy drift comparison:** the causal minimizer competes with every
   predictable budget-feasible dispatch/receipt policy with a necessary
   `2 e_k` estimation penalty.
4. **Finite-time learning/resource theorem:** the average stationarity measure
   is bounded by initialization, `O(1/V)` queue price, Markov/delay error,
   causal-cone error, and comparator remainder; the pathwise average budget
   violation is `O(V/K)` under the proved queue cap.
5. **Dynamic-over-static separation:** disjoint launch-state cones yield an
   `m`-fold edge-cost separation over any static graph with uniform zero cone
   error.

The still-open theorem interface is the executable critic estimate of signed
within-cone drift benefit and the full launch-to-receipt remainder under policy
changes during a random delay.

## Empirical package required for an ICML submission

1. A factored synthetic Markov game with known causal cones, where the theorem
   quantities and dynamic-over-static separation can be measured exactly.
2. Pistonball with 20 distinct actor blocks as the standard positive task,
   including conformal coverage, communication, sample count, update count,
   return, tail risk, and measured overhead.
3. HalfCheetah `6x1` as a dense control showing safe fallback rather than a
   fabricated sparse advantage.
4. Strong comparators: always-fresh complete graph, no-refresh/local, static
   physical graph, best fixed horizon/graph under the same budget, age-based
   asynchronous scheduling, and an oracle causal envelope used only as an
   upper bound.
5. Ablations separating causal cone, Lyapunov queue, receipt mass, delay,
   policy heterogeneity, and conformal policy-shift correction.

The main paper should lead with the positive mechanism and theorem.  Dense-task
fallback and estimator-development failures remain in internal provenance or a
single boundary experiment; they must not fragment the publication narrative.

## Immediate gates

1. Confirm the causal tube on fresh CPU calibration/holdout seeds.
2. Close the within-cone drift estimator and random-delay receipt remainder.
3. Demonstrate CPU synthetic controller headroom over the strongest static
   horizon/graph envelope.
4. Only then request GPU resources for Pistonball policy training.

Passing these gates would make the project a credible ICML candidate, not an
assurance of acceptance.  Failure of the causal-tube or strong-baseline gate
would require revising the central claim before GPU expenditure.
