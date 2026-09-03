# Arrival-time Lyapunov repricing for asynchronous constrained MARL

Status: **stopped by the bounded development feasibility scan**.  The exact
repricing identity is valid, but its broad incremental value over birth
pricing was too small.  See
`validation_lyapunov_repriced_async_cmdp_feasibility.md`.  This is not a
novelty claim, theorem, preregistration, or authorization for sampled/GPU work.

## One coherent question

Consider centralized training and decentralized execution in a cooperative
constrained Markov game.  Agent-owned policy-gradient packets finish at
heterogeneous times.  During a packet's flight, other policy blocks and the
team's shared constraint multiplier can both change.  A packet formed from a
single birth-time Lagrangian gradient is therefore stale in two distinct ways:

1. its policy argument is old; and
2. its safety/resource price is old.

The paper-level question would be:

> Can an event-driven trainer remove dual-price staleness exactly, control the
> remaining policy staleness, and attain the reward--feasibility quality of
> synchronous constrained MARL without waiting for the slowest actor?

Reward/constraint quality per charged transition is primary.  Wall-clock
throughput is a secondary but necessary Pareto axis, not the sole objective.

## Lyapunov is an online design state

For constraint costs `J_m^c(theta) <= d_m`, maintain physical virtual queues

```text
Q_m(k+1) = [Q_m(k) + observed_cost_m(k) - d_m]_+.
```

The current price is `lambda_m(k)=Q_m(k)/V`.  Instead of transmitting only the
already-combined birth-time Lagrangian gradient, packet `p` carries separate
reward and constraint components

```text
g_p^r, g_p^(c,1), ..., g_p^(c,M).
```

At arrival `k`, the server forms

```text
g_p^arrival = g_p^r - sum_m lambda_m(k) g_p^(c,m).
```

For the same birth-policy estimators, this removes dual-price delay exactly:

```text
g_p^arrival - g_p^birth
  = -sum_m [lambda_m(k)-lambda_m(b_p)] g_p^(c,m).
```

The queue therefore changes the executed actor direction, not merely a proof
constant.  A smoothness/staleness envelope may additionally determine a
closed-form packet scale.  The design remains `O(Md)` in packet dimension and
number of constraints; it has no agent-subset scan, Hessian inverse, or generic
QP.

## Why this is not a renamed stopped candidate

The stopped participation, refresh, graph, trust-radius, and parallel-commit
programs tried to adapt a stationary learning mechanism after a strong static
choice had already captured nearly all value.  Here the control state is the
actual accumulated team constraint debt.  If a constraint is active, its
price must change along training; a single static price cannot generally
represent both the safe transient and the constrained optimum.

This distinction does not establish novelty.  Generic asynchronous
primal--dual optimization, constrained MARL, Lyapunov safe RL, policy-lag
correction, and asynchronous actor--critic are all close boundaries.  The
candidate survives only if the factorized-policy packet decomposition yields a
new finite-time Markov-game result and standard constrained-MARL experiments
show a nontrivial Pareto improvement.  A fresh primary-source and citation
integrity audit is mandatory before manuscript use.

## Required theorem chain

1. Define a constrained Markov potential game and factorized CTDE policy; make
   every reward/cost trajectory and version dependency explicit.
2. Prove conditional unbiasedness/bias bounds for the decomposed Markov packet
   at its birth policy, and the exact dual-repricing identity above.
3. Use a composite Lyapunov function containing objective gap, physical
   constraint queues, and policy-delay history to prove a finite-time KKT/Nash
   residual and cumulative-violation bound for the same event-driven update.
4. Give a separation family in which birth-priced asynchronous primal--dual
   updates violate or oscillate, barrier synchronization is safe but
   throughput-limited, and arrival repricing obtains both feasibility and
   asynchronous progress.
5. State precisely what changes under neural approximation.  No confidence or
   monotonic-safety claim may be transferred from the finite/tabular theorem
   without proof.

An SDDE is optional.  It may characterize the small-step primal--dual delay
stability boundary, but the discrete event-time theorem is primary.

## Bounded survival gates

Before a sampled Markov experiment or GPU implementation:

1. an exact constrained-potential model must verify the repricing identity and
   exhibit a strict birth-price-delay failure;
2. on a declared active-constraint heterogeneous-delay population, causal
   repricing must reduce cumulative violation by at least 25% and composite
   finite-horizon risk by at least 10% relative to birth pricing in at least
   60% of cells;
3. it must remain within 10% of a current-price barrier method's composite risk
   while completing at least 20% more charged proposals per wall-clock horizon;
4. inactive-constraint controls must make birth and arrival pricing identical;
5. queue, trajectory, proposal, environment-step, and wall-clock accounting
   must all be explicit.

Failure of the exact identity or the broad active-constraint headroom gates
stops this candidate.  A favorable constrained benchmark cannot be chosen
after observing a failed analytic population.
