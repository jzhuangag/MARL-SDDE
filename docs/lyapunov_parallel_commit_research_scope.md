# Lyapunov parallel commit for asynchronous heterogeneous MARL

Status: **stopped after the frozen CPU oracle-headroom screen**.  The algebraic
interface remains valid, but the candidate failed every declared headroom gate
and is not authorized for sampled learning or GPU work.  See
`validation_lyapunov_parallel_commit_headroom.md`.

## One paper question

Sequential heterogeneous-agent policy optimization protects each actor's
surrogate from policy changes made by other actors, but serializes actor
optimization.  Computing all candidates in parallel removes that barrier, yet
committing all of them can violate the joint improvement bound.  The standard
Two Clocks pilot additionally showed that immediately committing more
self-fresh packets can worsen both AUC and terminal return even with equal
trajectory work.

The proposed question is:

> Can a centralized training-time Lyapunov commit layer turn asynchronously
> completed unilateral actor proposals into certified joint policy progress by
> choosing, in one low-complexity action, which proposals to commit and how far
> to move along each one?

Execution remains decentralized and unchanged.  Wall-clock efficiency is a
secondary consequence of parallel actor computation, not the sole objective.
The primary object is joint policy improvement under asynchronous candidate
availability.

## Joint action, not two heuristics

At commit event `k`, let `R_k` contain ready actor proposals.  For each ready
agent, `x_i in [0,1]` scales its base trust-region proposal.  The active set is
`{i:x_i>0}`.  Thus the online participation count and actor step sizes are

```text
q_k = number of nonzero x_i,
eta_(i,k) = eta_base_i x_i.
```

They are not selected in separate scans.

Assume the same data used to construct a proposal supplies a lower-confidence
linear gain `a_i`, while the joint performance lemma yields a positive
semidefinite diagonal-plus-rank-one curvature envelope

```text
H_k = diag(h_k) + rho_k w_k w_k^T.
```

The certified joint gain is

```text
G_k(x) = a_k^T x - (1/2) x^T H_k x.
```

The rank-one term represents the shared occupancy or joint importance-ratio
coupling that makes otherwise unilateral proposals interact.  A valid paper
must derive it from a declared finite Markov-game or trust-region performance
bound; it cannot be inserted only because it makes the optimizer convenient.

## Lyapunov is the design rule

Let `Q_i` be proposal-service debt and `Z` certificate-risk debt.  With risk
cost vector `c_k`, budget `epsilon_k`, and tradeoff `V`, use

```text
Q_i(k+1) = [Q_i(k) + A_i(k) - x_i(k)]_+,
Z(k+1)   = [Z(k) + c_k^T x_k - epsilon_k]_+.
```

The quadratic queue Lyapunov drift plus negative certified progress gives the
online problem

```text
maximize  (V a_k + Q_k - Z_k c_k)^T x
          - (V/2) x^T H_k x,
subject to 0 <= x_i <= 1 for ready proposals,
           x_i = 0 otherwise.
```

The same Lyapunov quantities therefore determine the algorithmic action and
the eventual finite-time bound.  They are not diagnostics added after
training.

For `H=diag(h)+rho w w^T`, the KKT equations reduce to

```text
x_i(z) = clip((ell_i - w_i z)/h_i, 0, cap_i),
z       = rho sum_i w_i x_i(z).
```

The right side is monotone.  Bisection gives the global box-QP optimum in
`O(n log(1/tolerance))` time and `O(n)` memory, with no subset enumeration,
Hessian inverse, preconditioner, or generic QP solver.  The executable solver
is `parallel_commit_qp.py`.

## Required theorem chain

All claims must hold for the same commit rule.

1. Derive the diagonal-plus-rank-one lower bound from a factorized-policy
   performance-difference argument, including Markov estimation error and
   proposal staleness.
2. Prove the scalar-root solution and a one-event composite drift inequality
   for objective gap, service debt, risk debt, and any required delay-history
   energy.
3. Establish finite-time potential stationarity or Nash-gap control plus
   average certificate-budget feasibility and all-agent service coverage.
4. Prove a separation family where sequential commits lose parallel compute,
   unrestricted simultaneous commits lose return, and the same Lyapunov QP
   obtains positive progress.
5. State the neural version as empirical unless its confidence and curvature
   envelope is actually certified.

An SDDE is optional and subordinate.  It is useful only if a small-step limit
of the primal/debt process yields an additional stability or phase theorem;
the discrete commit proof remains primary.

## Novelty and failure boundary

This is not the stopped scalar strategic-drift controller: the decision is a
vector and can change both concurrency and scale.  It is not the stopped
compatible-set MaxWeight scheduler: weakly interacting agents can receive
partial concurrent commits instead of being separated by a binary graph.  It
is not a revival of perishable-update backpressure: the queue price multiplies
the same joint policy-improvement bound optimized by the commit action.

The strongest reviewer objection is that this is HAPPO plus a generic convex
optimizer.  The idea survives only if the Markov-game derivation produces the
rank-one coupling, the Lyapunov queues give a non-asymptotic learning/coverage
result, and experiments beat both sequential HAPPO and a strong fixed global
trust-region scale.  A new literature and citation-integrity audit is required
before a novelty claim enters a manuscript.

## Bounded feasibility gates

Before any new GPU work:

1. exhaustive algebra must verify the QP solution and the exact performance
   lower bound on finite cooperative games;
2. a CPU analytic oracle screen must show at least 5% aggregate value over the
   stronger of best sequential order and best fixed global scale, with gains
   in at least 60% of declared heterogeneous cells;
3. a causal controller using only registered observable quantities must retain
   at least 80% of that oracle headroom;
4. service and risk debts must be nontrivial and stable, not inactive labels;
5. fixed work, independent confirmation seeds, strong baselines, overhead and
   failure controls must be specified before sampled confirmation.

Failure of either the performance-bound interface or CPU oracle-headroom gate
stops this candidate.  It must not be rescued by weakening comparators or
selecting a favorable standard task after outcomes.
