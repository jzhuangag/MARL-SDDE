# Validation of the outcome-free Two Clocks public-MPE bridge G0

Date: 2026-09-03.

## Decision

**Superseded by Amendment 1.**  The original G0 scientific-outcome exclusion,
ownership checks, and complete-packet accounting passed, but the frozen
barrier's unfinished tail was charged as one trajectory although every packet
contains two.  The original authorization is therefore withdrawn pending a
corrected frozen rerun.  This result contains no learning-performance evidence
and does not authorize a GPU run.

The frozen source commit was `b406e81`.  Primary and isolated reproduction
outputs were byte-identical with SHA-256
`d71d7c54ca76d2d0cbfab22189ffef787adba797c4ca9af8adbe9d1b6e810709`.

## Coverage

All 16 combinations of two public tasks, two service profiles, and four
execution methods ran on the pinned HARL commit.  Every mandatory invariant
passed:

- task shapes and agent counts matched the declared interface;
- initial policy parameters and past-data frozen control variates were paired
  across methods;
- all asynchronous packets were self-fresh in their owner block;
- complete packets used two independent trajectories and were fully charged;
- complete barrier packets were averaged under a frozen round policy;
- unfinished horizon-tail work was recorded;
- policy-motion diagnostics were finite and nonzero;
- the Lyapunov conditions were at most one; and
- no reward, return, success, advantage, gradient, or loss field was emitted.

Under the four-unit heterogeneous service smoke, `simple_spread_v2` completed
7 packets for every method.  The asynchronous methods applied 7 owner updates;
the barrier batched them into 3 owner updates.  For `simple_reference_v2`, all
methods completed 5 packets, with 5 asynchronous updates versus 2 barrier
updates.  These counts validate the intended clock difference; they are not a
learning result.

The theorem-derived neural multipliers were `0.575728` for the three-agent
task and `0.692810` for the two-agent task.  They were computed from the
declared interaction matrix and event-delay bound, rather than selected from
G0 data.

## Next authorized action

Create an independent pilot preregistration with fresh seeds, frozen source
and analyzer hashes, learning-curve checkpoints, complete physical-work and
policy-motion accounting, and the survival gates in
`post_controller_icml_mainline_decision.md`.  G0 internals must not be used to
select a task, learning rate, threshold, or comparator.
