# ICML asynchronous-MARL mainline closure

Date: 2026-09-05.

Status: **the present controller family is closed; no GPU escalation.**  This
is an internal research decision, not a claim that asynchronous MARL or
Lyapunov-designed learning is impossible.

## One conclusion from two independent kill gates

The project tested two ways to make Lyapunov theory an online design tool.

1. A coupled actor--critic action selected policy progress and critic repair
   by a two-variable drift QP.  Its exact-moment oracle had real value, but the
   frozen equal-cost ideal sensor was worse than the privileged online
   diagonal baseline: geometric AUC ratio 1.016520, only 32.91% of seed-cells
   improved, and exact coupling headroom recovery was -31.84%.  Coefficient
   error was small, so a better estimator is not the missing fix.
2. A compatible-update graph used Lyapunov MaxWeight to schedule independent
   sets of ready actor packets.  Its frozen outcome-free ceiling had
   dynamic/strong-compatible queue-cost ratio 1.026757, median reduction zero,
   and only 18.75% of scenarios with at least 10% reduction.  It improved
   throughput over sequential service by 7.61x, but did not improve the
   stronger compatible policy on the declared primary objective.

These failures have the same structure.  Relative to weak static or sequential
comparators, dynamic control looks valuable.  Relative to strong online
diagonal or ready-color comparators at matched resources, its remaining
learning-risk margin is too small.  The graph result's main value is
wall-clock throughput; the user explicitly does not want wall-clock to be the
only or primary scientific claim.

## Closed routes

The following must not be revived by changing grids, choosing favorable cells,
reducing the sensor charge, weakening comparators or assigning a new
experiment number:

- separately paid five-scalar drift sensing for joint `(alpha,beta)` control;
- generic compatible-set/independent-set MaxWeight scheduling;
- scalar stale-gradient damping or gradient-alignment control presented as a
  new MARL contribution;
- a dynamic collaboration graph whose only demonstrated advantage is service
  rate over fully sequential updates.

The high-target/low-noise drift-sketch subset is a valid descriptive phase,
not authorization for an outcome-selected main population.

## Retained publishable assets

The negative decision does not invalidate the following rigorously checked
objects:

- exact coupled actor--critic drift geometry and the target-motion reduction;
- the conditional finite-time tabular actor--critic theorem and corrected SPD
  critic step-size condition;
- the Markov packet decomposition and trajectory-level concentration
  interface;
- the nonvacuity lower audit showing why per-packet simultaneous safety is
  expensive;
- asynchronous block-gradient, wall-clock and slow-essential-agent lower
  bounds already recorded in the project;
- compatible scheduling capacity/throughput algebra as a secondary systems
  result;
- reproducible negative evidence against weak-baseline-only conclusions.

They can support a narrower theory/limits manuscript after a dedicated
novelty and venue audit.  They do not currently form an ICML-ready positive
algorithm paper.

## Conditions for an ICML research reset

A new ICML mainline must start from a genuinely different scientific question,
not another controller mutation.  Before algorithm construction it needs:

1. an intrinsic equal-resource learning-value gap of at least 10% against the
   strongest relevant online baseline on an outcome-free analytic family;
2. a control signal produced by mandatory training data or metadata, with no
   hidden sensing tax and no oracle state;
3. a theorem whose new object is specifically multi-agent and asynchronous,
   rather than delayed SGD, MaxWeight or adaptive step size with renamed
   variables;
4. a standard benchmark interface where the same mechanism can plausibly
   improve return/AUC at matched transitions and learner applications, with
   wall-clock reported as a secondary axis;
5. one coherent contribution contract connecting motivation, action,
   Lyapunov design, finite-time result and experiments.

No candidate satisfying all five conditions has been established in the
current bounded search.  The scientifically correct next action is therefore
to pause performance experiments, consolidate the theory/negative-bound paper
option, and conduct a fresh problem-level novelty search before allocating
GPU resources.  This is a stop decision for the present family, not a promise
that a future ICML idea cannot be found.
