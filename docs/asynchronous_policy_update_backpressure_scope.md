# Asynchronous policy-update backpressure: research scope

Date: 2026-09-01.
Status: Stage-1 research scope and feasibility contract.  This is not a
preregistration, an algorithm claim, a theorem claim, or experimental evidence.

## Research-question brief

### Primary question

Can an event-driven Lyapunov scheduler for perishable unilateral policy-update
proposals reduce wall-clock stationarity regret relative to both
barrier-synchronous and accept-all asynchronous training, while retaining a
finite-time convergence guarantee in cooperative Markov potential games with
heterogeneous Markovian proposal times?

### Why this is one problem rather than a list of mechanisms

In heterogeneous-agent CTDE, the actors are different policies in the same
game, not interchangeable workers estimating one global gradient.  A joint
rollout is collected under version `v`; each agent then computes a unilateral
actor proposal at its own speed.  When one proposal is deployed, the joint
policy and the state distribution change.  Every unfinished proposal therefore
becomes less valid.  Waiting for all proposals preserves a clean update order
but pays the slowest-agent barrier.  Applying every proposal removes that
barrier but lets fast agents make slow-agent proposals stale and can prevent
all-agent stationarity.

The object to control is consequently a queue of **perishable policy updates**:
serving one agent changes the value of the jobs still in every other agent's
queue.  The research problem is to schedule and scale these coupled updates in
wall-clock time.  Lyapunov drift is the decision rule for that scheduling
problem; it is not an analysis wrapper added after the algorithm.

### FINER assessment

| Criterion | Score | Reason |
|---|---:|---|
| Feasible | 4/5 | The first value screen is an exact CPU Markov-game simulation; a full neural CTDE benchmark will later need GPU. |
| Interesting | 5/5 | It exposes the throughput-versus-joint-nonstationarity conflict hidden by round-synchronous MARL training. |
| Novel | 3/5 | The exact combination is not present in the bounded source set, but the ingredients have strong close neighbors and the novelty claim is not yet established. |
| Ethical | 5/5 | No human data or deployment is involved. |
| Relevant | 5/5 | It targets wall-clock training of heterogeneous cooperative agents without changing decentralized execution. |
| **Average** | **4.4/5** | Conditional on the two feasibility gates below. |

### Scope boundaries

In scope:

- fully cooperative discounted Markov games and Markov potential games;
- distinct agent policies and agent-wise actor updates under CTDE;
- random, heterogeneous completion times for trajectory processing or actor
  optimization;
- Markovian trajectory noise, delayed proposals and joint-policy drift;
- a centralized training-time scheduler with decentralized execution;
- wall-clock stationarity regret, return, proposal freshness and all-agent
  update coverage.

Out of scope:

- federated workers that all train a single common policy;
- participation-count selection, communication-graph learning, or selecting
  only `q` and a learning rate;
- asynchronous action or macro-action execution as the primary problem;
- external safety constraints introduced merely to manufacture dynamic value;
- unrestricted general-sum games, arbitrary unbounded delays and a neural
  global-convergence claim;
- SDDE as a required proof device.  A delay-equation interpretation may be
  reported only if it adds a result not already given by the event-time proof.

Execution-time communication is unchanged from the underlying MARL policy.
The proposed scheduler exists only during centralized training.

### Sub-questions

1. What observable event-time quantity upper-bounds the loss of validity of a
   unilateral proposal after other agents update?
2. Can one composite Lyapunov function simultaneously control optimization
   progress, pending-proposal perishability and starvation of slow agents?
3. Does the resulting causal scheduler have material wall-clock value over a
   strong, per-scenario tuned family of synchronous and asynchronous rules?

## Model and candidate mechanism

Let the joint policy be `pi_theta = product_i pi_{theta_i}` and let `Phi(theta)`
denote the cooperative return or Markov potential.  Agent `i` starts a proposal
from joint-policy version `tau` and returns an estimated unilateral direction
`u_(i,tau)` at event `k`.  Its displacement from the policy that generated its
data is

```text
d_(i,k) = theta_k - theta_tau.
```

Each accepted update changes `d_(j,k)` for every still-pending proposal `j`.
The lifted event state is therefore the current joint policy together with the
birth version, age and direction of every pending proposal.  It is not enough
to retain a scalar timestamp.

A candidate composite Lyapunov function is

```text
L_k = Phi_star - Phi(theta_k)
      + (lambda / 2) sum_(j pending) ||d_(j,k)||^2
      + (beta / 2) sum_i Q_(i,k)^2,
```

where `Q_i` is update-service debt for agent `i`.  It grows with wall-clock time
when coordinate `i` is not given a valid update and is served only by an
accepted, certified proposal.  This term is necessary because convergence to
an all-agent stationary point cannot follow if a strategically important slow
agent is starved.

For a completed proposal `i`, a smoothness and stale-gradient bound should
yield a causal lower certificate of the form

```text
DeltaPhi_i(alpha) >= alpha * Ghat_i
                    - c_curv * alpha^2 * ||u_i||^2
                    - c_stale * alpha * ||u_i|| * ||d_(i,k)||
                    - eps_markov(i,k).
```

The action also increases the perishability term of every other pending
proposal:

```text
||d_j + alpha u_i||^2 - ||d_j||^2.
```

At each completion event, the scheduler maximizes the negative upper bound on
the conditional Lyapunov drift over `accept/refresh` and scalar `alpha`.  After
norm-bounding the cross terms, the online subproblem is a concave scalar
quadratic and has a clipped closed form.  Choosing among simultaneously ready
agents is a linear scan.  The intended controller cost is `O(nd)` arithmetic,
`O(nd)` pending-state memory and no Hessian, covariance matrix or preconditioner.

The name **Policy-Update Backpressure (PUB)** is provisional.  It denotes the
single scheduling mechanism above, not a bundle of unrelated contributions.

## Theorem programme

The following are obligations, not completed theorems.

1. **Stale unilateral proposal lemma.**  For a smooth cooperative Markov
   potential, bound the conditional proposal bias by the joint-policy path
   since its rollout plus a geometrically-mixing Markov estimation term.
2. **One-event drift lemma.**  Bound the drift of the exact lifted Lyapunov
   state under accept, scale and refresh actions.  Every wall-clock and rollout
   cost must appear in this event-time inequality.
3. **Finite-time stationarity and service theorem.**  Under bounded second
   moments, a mixing envelope and a strict feasibility condition for proposal
   service, show a wall-clock weighted stationarity bound and sublinear
   all-agent service deficit.  A target form is

   ```text
   sum_k E[Delta t_k ||grad Phi(theta_k)||^2] / sum_k E[Delta t_k]
     <= O(1/V) + O(V/T_wall) + eps_mix + eps_cert,
   ```

   with a corresponding bound on `sum_i E[Q_i]`.  Selecting `V` as a function
   of the horizon is analysis, not offline tuning of per-agent step sizes.
4. **Separation theorem.**  Exhibit one coupled two-agent potential game and
   stochastic completion process where a barrier policy pays the slowest-agent
   clock and accept-all incurs nonvanishing stale-coordinate error, while a
   causal backlog policy achieves vanishing wall-clock stationarity regret.

The proof is not allowed to replace the executable controller with an oracle,
assume independent gradients after conditioning on a shared Markov trajectory,
or infer convergence merely from bounded virtual queues.

## Experimental methodology blueprint

### Stage A: problem-value screen (CPU, outcome-free)

Use exact finite cooperative Markov games with three heterogeneous agents.
Enumerate policies exactly and drive agent proposal completions with declared
Markov-modulated latency traces.  Compare a feasible non-myopic schedule and
the executable PUB rule against the stronger of:

- barrier-synchronous fresh sequential update;
- simultaneous batch update;
- accept-all asynchronous update;
- best fixed trust radius;
- best fixed age/path-length decay;
- best fixed refresh cadence and fixed staleness threshold.

Every static hyperparameter is selected per development scenario.  Report both
absolute return and wall-clock regret to the exact best factorized policy.
The two mandatory value gates are:

1. median feasible dynamic reduction in wall-clock regret at least 10%;
2. reduction at least 5% in at least 60% of declared cells.

Both must pass before an algorithm-efficacy pilot is authorized.  A positive
return comparison against a weak accept-all baseline is insufficient.

### Stage B: independent CPU confirmation

Only after Stage A and the first three proof obligations pass: freeze the
implementation, scenario population, costs, gates and fresh confirmation
seeds.  Include Markov mixing, coupling strength, latency ratio/burstiness,
number of agents, proposal noise, ablations of each Lyapunov term and measured
controller overhead.  Development outcomes cannot select confirmation cells.

### Stage C: standard cooperative MARL

Only after independent CPU confirmation: implement PUB in the official HARL
training stack and evaluate on heterogeneous-agent Multi-Agent MuJoCo and at
least one second benchmark such as SMACv2.  Baselines must include HAPPO/HATRPO
or their current HARL equivalents, MAPPO, a barrier-parallel implementation,
accept-all async, a staleness-adaptive rule and an AFedPG-style delay correction
adapted without pretending workers are game agents.  Report environment steps,
gradient updates and measured wall-clock separately.  This stage is expected
to require GPU/HPC4 and is not yet authorized.

## Bounded novelty confrontation

- HARL/HAPPO proves monotonic improvement by sequential agent-wise updates and
  permits randomized order, but its algorithmic contract is round based and
  does not schedule concurrently generated perishable proposals in wall-clock
  time: [JMLR 2024](https://jmlr.org/papers/v25/23-0488.html).
- AFedPG analyzes stale asynchronous policy gradients from heterogeneous
  federated workers that collaborate on one global policy; those workers are
  not distinct actors whose unilateral changes alter one another's game
  gradients: [ICLR 2025](https://openreview.net/pdf?id=5DUekOKWcS).
- asynchronous actor-critic and ACAC address agents whose **actions** or
  macro-actions finish at different times, not actor optimizers whose policy
  proposals finish at different times: [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1c153788756d35559c22d105d1182c30-Abstract-Conference.html),
  [ICML 2025](https://proceedings.mlr.press/v267/jung25a.html).
- asynchronous decentralized actor-critic with a Markov near-potential uses an
  approximate Lyapunov function for asymptotic learning dynamics; it does not
  optimize a centralized CTDE proposal schedule or wall-clock service debt:
  [arXiv:2409.04613](https://arxiv.org/abs/2409.04613).
- staleness-adaptive trust regions already couple clipping to asynchronous
  single-policy mismatch.  PUB must therefore contribute multi-agent
  cross-perishability and scheduling, not merely another age-dependent step
  size: [arXiv:2607.18722](https://arxiv.org/abs/2607.18722).
- safe/constrained MARL already combines local policy updates and constraints.
  External safety is deliberately not the source of PUB's dynamic value:
  [Scal-MAPPO-L](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fa76985f05e0a25c66528308dda33de0-Abstract-Conference.html).

This bounded scan makes the candidate plausible, not novel by declaration.  A
fresh search and citation-integrity pass is required before manuscript use.

## Devil's-advocate checkpoint 1

Verdict: **REVISE BEFORE FREEZE.**  No critical logical contradiction is known,
but four major risks block an ICML claim today.

1. The controller can collapse to a generic asynchronous coordinate-descent
   scheduler if the Markov-game performance lemma contributes no genuinely
   multi-agent cross term.
2. A best fixed staleness rule may remove most wall-clock headroom, as happened
   in the preceding stationary policy-backpressure audit.
3. The service-debt requirement may look artificial unless it is tied exactly
   to all-coordinate stationarity and is ablated against unbiased random agent
   selection.
4. Neural PPO ratios do not provide a uniform full-policy distance certificate;
   the practical proxy and the theorem-facing quantity must be separated.

The strongest likely reviewer objection is: “This is MaxWeight scheduling plus
a stale-gradient bound, evaluated under injected delays.”  The candidate
survives that objection only if the separation theorem, a multi-agent
cross-perishability term and broad standard-benchmark wall-clock gains all use
the same executable mechanism.

## Decision rule

The candidate is worth one bounded theory/value investigation.  It is not yet
an ICML-ready mainline.  Failure of either the performance-bound interface or
the CPU oracle headroom gates permanently stops this construction; it must not
be rescued by weakening baselines, changing the metric after outcomes, or
adding an unrelated constraint.

