# Beyond Sequential MARL: Lyapunov Scheduling of Compatible Asynchronous Policy Updates

Status: substantive successor premise and outcome-free algebra only.  No
efficacy experiment, formal seed or GPU benchmark is authorized.

## The single research problem

Heterogeneous-agent trust-region methods obtain monotonic joint-policy
improvement by updating actors sequentially.  Sequentiality prevents one
agent's policy change from invalidating the next agent's improvement surrogate,
but it also creates a training barrier whose wall-clock cost grows with agent
count and heterogeneous actor/learner times.  Naive simultaneous updates remove
the barrier but can combine individually useful actor steps into a harmful
joint step.

The proposed question is:

> Which asynchronously ready actor-policy updates can be committed together,
> so that sparse interaction permits parallel wall-clock progress without
> giving up a certified joint-improvement bound?

This changes the task-level decision.  The algorithm does not choose a scalar
participation count, graph weight, clipping threshold or correction strength.
It schedules a compatible set of distinct actor updates from asynchronous
ready queues.  Training is centralized and asynchronous; execution remains
fully decentralized and unchanged.

## Compatible-update geometry

Assume a cooperative objective has the block lower-smoothness bound

\[
J(\theta+u)-J(\theta)
\ge \sum_i\langle\nabla_iJ(\theta),u_i\rangle
-\frac12\sum_{i,j}L_{ij}\|u_i\|\|u_j\|,
\tag{1}
\]

with a symmetric nonnegative interaction matrix `L`.  For certified block
directions `gtilde_i`, radii `r_i`, signals `s_i=||gtilde_i||` and
`u_i=alpha_i gtilde_i`, (1) gives

\[
\mathcal G(S,\alpha)
=\sum_{i\in S}\alpha_i(s_i^2-r_is_i)
-\frac12\sum_{i,j\in S}L_{ij}\alpha_i\alpha_js_is_j.
\tag{2}
\]

Define a conflict graph containing edge `(i,j)` whenever the certified
cross-block constant `L_ij` is nonzero or exceeds a declared approximation
threshold.  If `S` is an independent set in the exact graph, the off-diagonal
part of (2) vanishes and the certified gains are additive.  Two updates that
are each beneficial can be jointly harmful when they share an edge; the
repository algebra contains an explicit witness.

For a general Markov game, sparsity is not inferred merely from a local reward
diagram.  The theorem needs a causal-cone or decay-of-correlation result showing
that distant actor-policy changes have zero or controlled effect on the
discounted occupancy and advantage.  A public conservative graph is allowed;
learning a graph and then claiming coverage from the same data is not.

## Asynchronous queues and Lyapunov scheduling

Let `Q_i(k)` be the number or service deficit of ready certified proposals for
actor `i`.  At a learner scheduling epoch, choose a compatible ready set `S_k`.
With arrivals `A_i(k)` and unit service,

\[
Q_i(k+1)=[Q_i(k)-1\{i\in S_k\}]_+ + A_i(k).
\tag{3}
\]

Use the queue Lyapunov function

\[
\mathcal Q_k=\frac12\sum_iQ_i(k)^2
\tag{4}
\]

and select the independent set maximizing

\[
\sum_{i\in S_k}
\left[Q_i(k)+V\,g_i^{\rm cert}(k)\right],
\tag{5}
\]

where `g_i^cert` is the best certified single-block gain from (2).  Equation
(5) is the drift-plus-progress decision: backlog prevents heterogeneous slow
actors from starving, while certified gain prioritizes useful ready work.

On path/tree conflict graphs, maximum-weight independent-set scheduling is
exact in linear time by dynamic programming.  On bounded-degree general
graphs, a deterministic maximal-weight greedy rule is linear-logarithmic and
must carry its approximation factor into the throughput theorem.  No dense
Hessian, inverse, preconditioner or QP is required.

## Required theorem chain

The candidate survives only if a single analysis proves:

1. a causal interaction graph or explicit weak-edge remainder for the declared
   Markov-game class;
2. simultaneous independent-set monotonic improvement from the same estimator
   used by the algorithm;
3. queue stability for heterogeneous Markov arrivals inside the compatible-set
   capacity region, including approximation factors;
4. finite wall-clock stationarity using stable service to guarantee actor-block
   coverage;
5. a strict speed separation from sequential HAPPO/HATRPO-style updates and a
   stability separation from unrestricted simultaneous updates;
6. complexity and communication accounting for graph construction, proposal
   certification and scheduling.

An exact sparse theorem may use factored contextual games or finite causal
cones.  A standard benchmark claim requires a separately proved truncation or
decay bound; visual proximity or an attention map is not a certificate.

## Novelty boundary

Sequential heterogeneous-agent policy optimization and monotonic-improvement
theory are inherited from HATRPO/HAPPO/HAML.  Parallel sequence-model training
in MAT, sparse action-dependency graphs, localized constrained MARL,
asynchronous action execution, generic MaxWeight scheduling and parallel block
coordinate descent are also inherited.  The candidate novelty requires their
specific missing bridge: certified compatible **training-update concurrency**
for distinct asynchronous actor queues, plus a joint learning/throughput
theorem and benchmark implementation.

Primary boundaries checked at design time include:

- [Heterogeneous-Agent Reinforcement Learning (JMLR 2024)](https://jmlr.org/papers/v25/23-0488.html);
- [Multi-Agent Transformer (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html);
- [Scalable Constrained Policy Optimization for Safe MARL (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fa76985f05e0a25c66528308dda33de0-Abstract-Conference.html);
- [Sparse Action-Dependent Policy Iteration (UAI 2026)](https://proceedings.mlr.press/v337/ding26a.html).

## Evidence gate

Before any sampled learner, run one outcome-free scheduling ceiling on path,
tree, grid and clustered conflict graphs with deterministic Markov-modulated
ready traces.  It must show material throughput/latency value over sequential
updates **and** strong fixed-color compatible scheduling while maintaining
coverage.  Passing only versus the sequential baseline is insufficient.

Only after the graph theorem and scheduling ceiling both pass may a disjoint
CPU stochastic potential-game pilot be preregistered.  Standard cooperative
MARL and GPU work remain later gates.

