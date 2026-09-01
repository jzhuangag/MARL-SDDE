# Stage-1 checkpoint: perishable policy updates in asynchronous MARL

Date: 2026-09-01.
Status: research-question and theorem-feasibility checkpoint.  This is not a
paper claim, preregistration or experimental success report.

## Unified research question

In cooperative CTDE with distinct agent policies and heterogeneous optimizer
completion times, can a causal stochastic-optimization rule schedule and scale
unilateral policy proposals whose validity is consumed by other agents'
updates, while attaining finite-time stationarity and lower wall-clock regret
than barrier and accept-all training in the identifiable high-load phase?

The source of dynamic value is endogenous: applying one agent's update changes
the joint occupancy and the gradients of pending agents.  It is not an
external safety constraint, a varying reward, a participation-count problem or
multiple workers estimating one shared policy.

The proposed paper title is provisional:

> **When Policy Updates Expire: Lyapunov Backpressure for Asynchronous
> Multi-Agent Reinforcement Learning**

## Single algorithmic object

For completed proposal `g_hat_i`, define cross-agent freshness debt `Z_i`,
sampling radius `r_i` and outgoing debt price

```text
P_i = sum_(j != i) C_(j,i) Z_j.
```

The dimensionless coordination load is

```text
Lambda_i = (Z_i+r_i)/||g_hat_i|| + P_i/(V||g_hat_i||).
```

`Lambda_i<1` is the identifiable phase.  Minimizing one Lyapunov drift bound
gives the closed-form step

```text
alpha_i = clip(
  V ||g_hat_i||^2 (1-Lambda_i)
  / [||g_hat_i||^2(VL_i+sum_j C_(j,i)^2)],
  0,
  alpha_bar
).
```

The same expression performs admission (`alpha_i=0`), online step selection
and ready-proposal ordering.  Runtime is a linear scan over pending agents and
does not use a Hessian inverse, covariance matrix, preconditioner or offline
learning-rate catalogue.

## Theory completed at this checkpoint

The proof document establishes, under explicit smoothness, coverage and
completion assumptions:

1. a proposal-gain lower certificate and the expiry boundary;
2. exact cross-agent debt dynamics;
3. the one-event Lyapunov drift and closed-form minimizer;
4. a low-load theorem showing a fixed step retains at least
   `(1-epsilon)^2` of the fresh quadratic certificate;
5. a high-load finite-horizon construction where barrier and like-for-like
   accept-all incur `Omega(M)` wall-clock regret while the Lyapunov rule incurs
   `O(1)`;
6. a finite-time full-gradient stationarity bound, with all-agent completion
   gap `D`, of order `O((D+nD^2)/K)` for exact gradients and a declared
   sampling-radius floor for stochastic proposals;
7. an episodic on-policy radius from independent bounded trajectory-gradient
   estimates without treating transitions inside one trajectory as
   independent.

The algebra audit contains 144 exact stale-bias checks, 576 gain checks, 144
closed-form optimizer checks, 405 finite-time event checks and two exact
high-load witnesses.  These tests check formulas; they are not experimental
evidence.

Targeted validation is `14 passed`; the established full regression is
`874 passed, 7 skipped`.  A bare root `pytest` command is not a valid regression
in this repository because preserved `tmp/t021_*` source snapshots duplicate
test module names; the execution record retains that collection-only failure
instead of deleting provenance.

## Evidence and negative result retained

The earlier 144-cell exact development scan failed its universal gates:

- median dynamic headroom: `7.2183%` versus required `10%`;
- cells reaching `5%`: `54.1667%` versus required `60%`.

Beam widths 128 and 256 and a retrospective best-width portfolio also failed.
The universal-controller claim remains stopped.  The result is consistent with
the low-load theorem: weak coupling has almost no dynamic headroom, while
strong coupling and persistent heterogeneity show a descriptive phase signal.
Those old cells cannot become confirmation evidence.

## Remaining feasibility risks

1. **Executable cross sensitivity.**  Exact finite games can compute
   `C_(j,i)`; neural policies need either a conservative analytic bound or a
   fully charged estimator with coverage.  A generic PPO ratio is not such a
   certificate.
2. **Continuing Markov data.**  The episodic bounded-estimator case is closed.
   A continuing-trajectory implementation must instantiate a verified
   mixing/spectral-gap radius and critic/off-policy bias terms.
3. **Empirical nontriviality.**  The exact witnesses prove the high-load region
   is nonempty; a disjoint CPU population must show the executable rule accepts
   useful updates and clears predeclared headroom gates against strong static
   comparators.
4. **Neural translation.**  No standard MARL GPU benchmark is authorized until
   the CPU phase confirmation passes.

## Stage-1 decision

**CONDITIONAL GO FOR ONE PHASE-CONDITIONED CPU STUDY; NO GPU AND NO PAPER CLAIM
YET.**

The next study may be designed only from the analytic load `Lambda`, not from
filtering favorable old outcomes.  It must:

- create a disjoint exact Markov-potential-game family with analytically
  declared low, transition and high-load strata;
- implement the actual closed-form controller, not the non-myopic beam;
- include per-cell tuned fixed steps, age/path decay, direct-TV gates,
  barrier, fresh serial, accept-all and a staleness-adaptive trust-region
  comparator;
- freeze controller, costs, strata, seeds, hashes and gates before outcomes;
- require positive high-load headroom, low-load near-ties, correct directional
  phase ordering, all-agent stationarity, no budget omission and measured
  controller overhead;
- stop before neural/GPU work if any mandatory gate fails.

This checkpoint advances a coherent ICML candidate, not an acceptance claim.
The main risk has moved from basic theorem feasibility to whether conservative
observable debt retains enough value in a disjoint executable experiment.
