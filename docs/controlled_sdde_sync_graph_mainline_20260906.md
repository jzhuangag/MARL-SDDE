# Controlled-SDDE policy-dependency synchronization for asynchronous MARL

Date: 2026-09-06

Status: problem and theorem design; authorizes only a separately frozen exact
CPU oracle-headroom gate.  It is not an efficacy result, a theorem completion,
or an ICML-readiness claim.

## 1. One research question

In asynchronous centralized training with decentralized execution (CTDE), can
a learner use a controlled stochastic delay differential equation (SDDE) and
a composite Lyapunov function to decide which teammate policy versions each
actor-specific rollout worker must refresh, so that joint-policy learning
retains most of the benefit of fresh trajectories under a long-run
communication budget?

The paper-level object is **pairwise strategic freshness**.  It is not worker
participation, parameter averaging, execution-time message attention, or the
recovery of one arbitrary synchronous optimization path.

## 2. Architecture and meaning of the graph

There are `n` distinct policy blocks

\[
\theta=(\theta_1,\ldots,\theta_n)
\]

in one cooperative Markov game.  A rollout worker owned by policy block `i`
generates a joint trajectory using its current owner block and cached teammate
blocks

\[
\chi_i=(\theta_i,\chi_{1\to i},\ldots,\chi_{n\to i}).
\]

Single-flight ownership makes the owner block self-fresh.  Teammate caches can
be strategically stale.  Before launching the next packet for owner `i`, the
learner chooses a predictable directed refresh set `S_i`.  An edge `j -> i`
means that the newest `theta_j` is transmitted to worker `i`, setting
`chi_(j->i) = theta_j`; it never copies `theta_j` into `theta_i`.

Training is distributed and asynchronous.  The centralized training service
knows policy versions, packet birth times, and the refresh ledger.  Execution
uses only the learned local policies and introduces no new communication
requirement.

The primary resource is trajectory/sample progress under a fixed message
budget.  Wall-clock time is a secondary systems metric rather than the
scientific estimand.

## 3. Exact event model before any diffusion approximation

Let `I_k=i` be the owner of the packet completed at learner event `k`, and let
`b_k` be its launch event.  The update is

\[
\theta_i^{k+1}=\theta_i^k-\alpha_{i,k}\widehat g_{i,k}(\chi_i^{b_k}),
\qquad
\theta_j^{k+1}=\theta_j^k\quad(j\ne i).
\]

For a smooth Markov-potential loss `f`, assume the block interaction envelope

\[
\|\nabla_i f(x)-\nabla_i f(y)\|
\le L_{ii}\|x_i-y_i\|+
\sum_{j\ne i}L_{ij}\|x_j-y_j\|.
\]

The launch-time residual strategic-staleness bound is

\[
B_{i,k}(S_i)=
\sum_{j\notin S_i}L_{ij}
\|\theta_j^{b_k}-\chi_{j\to i}^{b_k}\|+
B^{\rm flight}_{i,k}+B^{\rm Markov}_{i,k}.
\]

Every term must be predictable when the refresh set is committed.  The first
term uses mandatory parameter-version metadata.  The flight term pays motion
during packet service.  The Markov term must come from a Poisson-equation or
regenerative-block argument; calling a trajectory innovation independent is
not allowed.

## 4. Controlled hybrid SDDE

After a declared small-step scaling, the state consists of policy parameters,
directed caches, packet-age/history variables, and the communication queue.
Between cache refreshes, a candidate limiting model is

\[
d\theta_i(t)=
-u_i(t)g_i(\theta_i(t),\chi_{-i\to i}(t))dt
+\sqrt{u_i(t)}\Sigma_i(\theta(t),\chi(t))dW_i(t),
\]

with delayed arguments induced by packet service.  A selected refresh edge is
a reset jump

\[
\chi_{j\to i}(t^+)=\theta_j(t).
\]

This is a controlled hybrid SDDE, not an ordinary memoryless SDE.  Its
generator therefore contains continuous Itô terms, delayed history terms, and
the jump contribution

\[
r_{ij}a_{ij}(t)
\{\mathcal V(R_{ij}z_t)-\mathcal V(z_t)\}.
\]

The continuous model is a design and phase-analysis layer.  The paper still
requires an exact discrete event-time theorem and a finite-horizon weak-error
or coupling result connecting it to the SDDE.  Simulation agreement is not an
approximation theorem.

## 5. Lyapunov function and online decisions

Use

\[
\mathcal V_t=
f(\theta_t)
+\frac12\sum_{i}\sum_{j\ne i}p_{ij}
\|\theta_j(t)-\chi_{j\to i}(t)\|^2
+\frac{\rho}{2}\mathcal H_t
+\frac12 Q_t^2.
\]

Here `H_t` is a Lyapunov--Krasovskii history energy for in-flight packets, and

\[
Q_{k+1}=[Q_k+c(S_k)-\bar c]^+
\]

is the communication virtual queue.  The four terms have one interpretation:
current learning loss, pairwise strategic-staleness debt, service-delay debt,
and resource debt.

Refreshing `j -> i` removes the currently observed edge mismatch from the
second term.  Applying a packet changes the first and history terms.  A smooth
descent calculation gives the theorem-facing upper-bound shape

\[
\begin{aligned}
\mathbb E_k[\Delta\mathcal V]
\le{}&-\alpha_{i,k}\|\nabla_i f(\theta^k)\|^2
+\alpha_{i,k}\|\nabla_i f(\theta^k)\|B_{i,k}(S_i)\\
&+C_i\alpha_{i,k}^2(\sigma_i^2+B_{i,k}(S_i)^2)
-\sum_{j\in S_i}\frac{p_{ij}}2
\|\theta_j-\chi_{j\to i}\|^2\\
&+Q_k(c(S_i)-\bar c)+R^{\rm flight}_{i,k}.
\end{aligned}
\]

This display is a design contract, not yet a proved theorem.  In particular,
the reset benefit and future packet benefit occur at different event times;
`H_t` and the packet pipeline must close that timing gap.

The executed variables are:

1. launch-time refresh edges `S_i`;
2. receipt-time scalar update mass `alpha_(i,k)`.

For a fixed candidate update mass, additive certified edge benefits and linear
message prices make the exact graph decision a thresholded sort, not a generic
matrix QP.  Joint minimization can be performed by sorting the finitely many
edge breakpoints and solving the scalar convex quadratic for `alpha` on each
interval.  With a sparse declared dependency envelope, this costs
`O(|E_i| log |E_i|)` per launch and `O(dim(theta_i))` for the ordinary policy
update.  No Hessian inverse or dense covariance matrix is required.

## 6. Intended theorem chain

The project proceeds only if the following chain closes without oracle
quantities in the executable rule.

1. **Predictable Markov packet lemma.**  Establish bias, conditional variance,
   and norm bounds for a trajectory launched from a mixed-version joint policy.
2. **Discrete composite-drift theorem.**  Prove the event-time inequality for
   the actual launch/receipt filtration, including cache resets and in-flight
   packets.
3. **Budget and stationarity theorem.**  Show mean-rate stability of `Q` and an
   `O(T^{-1/2})`-type potential-stationarity bound plus explicit Markov,
   approximation, and communication terms.
4. **Drift-regret theorem.**  Bound cumulative certified drift relative to the
   best causal budget-feasible refresh policy, not merely a static graph.
5. **Static-graph separation.**  Give a switching sparse-interaction subclass
   where every fixed graph under the same communication budget has a positive
   gap, while the dynamic rule tracks the active dependency.
6. **Controlled-SDDE consistency.**  State a scaling regime and prove the
   discrete interpolation converges weakly to the hybrid SDDE over a finite
   horizon; use functional Itô/Dynkin analysis only within that regime.

The main theorem should combine items 2--4.  The SDDE is meaningful only if
item 6 is proved; otherwise the discrete theorem remains primary and the SDDE
claim is removed.

## 7. Why this is not the closed dynamic-collaboration route

The T-070--T-083 graph mixed donor estimates according to an unknown,
time-varying affinity.  Its observable controller had to identify transfer
quality from noisy residual fingerprints; paid probes and an invalid repeated
certificate caused the eventual failure.

The present graph transmits exact policy versions to rollout workers.  Its
basic mismatch `theta_j - chi_(j->i)` and packet age are known without an
extra environment transition.  The algorithm does not claim that one agent's
sample is an unbiased sample for another agent and does not mix personalized
parameters.  Thus it removes the old hidden-affinity identification problem.

It also differs from strategic-clock equalization.  The objective is the
Markov-game potential under a message budget, not closeness to a chosen
synchronous discretization.  The strongest comparator must include fixed
edge-specific refresh rates and staleness-only online scheduling.

## 8. Boundary against existing work

Yu, Chen, and Poor optimize active worker number, group size, staleness
thresholds, and communication protocols for workers updating one shared SGD
parameter through an SDDE characteristic-root analysis.  They do not have
distinct policy blocks, pairwise teammate caches, or an online policy-
dependency graph.

Min et al. study asynchronous communication among parallel agents solving the
same linear MDP through a central server.  Xiao et al. study asynchronous
environment action durations.  Dynamic coordination graphs and learned MARL
communication methods change value factorization or execution messages.
HyperMARL and adaptive parameter-sharing methods address gradient interference
architecturally.  None of these distinctions alone establishes novelty; the
complete cache-state/controlled-SDDE/discrete-drift theorem and empirical
return--communication frontier must survive a fresh search before submission.

Verified source metadata and read boundaries are recorded separately in
`controlled_sdde_sync_graph_sources_20260906.json`.

## 9. Mandatory problem gates before a controller pilot

An outcome-free exact linear-quadratic potential-game audit must use the
registered causal one-step drift policy as its optimistic dynamic ceiling and
compare that ceiling with all of:

- no refresh;
- periodic full refresh at the same average message cost;
- every feasible fixed directed graph;
- the best fixed edge-specific refresh rates;
- online largest-parameter-mismatch refresh;
- online oldest-cache refresh;
- online active-edge oldest-cache refresh.

Across a frozen grid containing stationary, switching, sparse, dense, weak-
coupling, strong-coupling, balanced-clock, and heterogeneous-clock controls,
all of the following are mandatory:

1. at least 10% median equal-resource oracle improvement over the strongest
   non-oracle causal baseline on the active population;
2. strict improvement in at least 60% of active cells;
3. no more than 1% loss in stationary and uncoupled controls;
4. positive headroom for nonzero temporal correlation and every registered
   delay group;
5. graph changes in at least 50% of active cells;
6. at least 90% of the oracle gain remains after charging every refresh;
7. no use of T-083A or strategic-clock outcomes for selection;
8. byte-identical reproduction before any sampled pilot.

Failure stops this architecture; thresholds, baselines, and populations may
not be changed after seeing the result.

## 10. Devil's-advocate gate

Verdict: **REVISE BEFORE THEOREM FREEZE; GO ONLY TO THE EXACT ORACLE GATE.**

The strongest objections are:

1. a centralized learner may already distribute a complete joint-policy
   snapshot, making pairwise refresh artificial unless the rollout-worker
   deployment and byte costs are implemented faithfully;
2. unknown `L_ij` values can turn the practical rule into an uncertified
   heuristic; a theorem must use public bounds or a separately priced
   estimator;
3. minimizing a local drift bound can be conservative and need not improve
   deep-policy return;
4. the hybrid reset process and trajectory delay make a valid SDDE limit much
   harder than writing a formal stochastic differential equation;
5. fixed-rate edge schedules may eliminate most apparent dynamic value, as a
   fixed scalar scaling eliminated strategic-clock headroom.

The exact oracle gate tests objection 5 first because it is the cheapest fatal
test.  Theorem construction is not authorized if that gate fails.
