# LCO-U0 perfect-observation controlled-sensing upper bound

## Purpose

Two causal hidden-geometry controllers have failed their frozen development
gates.  LCO-U0 is the resulting life-or-death feasibility audit.  It does not
test another controller and uses no trajectory seed.  Instead it asks whether
*any* causal policy with the same action-dependent sensing interface can have
enough adaptation value to justify more algorithm design.

The audit deliberately strengthens the sensor.  A plain update reveals
nothing; a fully charged optimistic update reveals the current potential or
rotational phase perfectly, but only after the current action.  The hidden
phase follows the declared two-state Markov law.  This perfect paid sensor is
strictly more informative than the noisy directional fingerprint, so its
optimal performance is an upper bound on the attainable fingerprint
performance.

## Exact finite-state belief representation

After a perfect observation, the belief is determined by the observed phase
and its age.  For stationary rotation probability \(f\), persistence
\(\rho\), last observed phase \(h\in\{0,1\}\), and age \(a\),

\[
p(h,a)=f+\rho^a(h-f).
\]

The audit uses age truncations 128, 256, and 512.  At the truncation boundary a
no-call transition returns to the stationary no-observation belief rather than
creating a spurious recurrent state.  Convergence between ages 256 and 512,
flow balance, normalization, phase calibration, and resource use are mandatory
numerical gates.

For each game cell, a sparse average-cost occupation-measure linear program
minimizes expected certified log multiplier subject to the same average
optimism budget.  Every call both applies optimism and generates the perfect
observation.  The comparison set contains:

- the best period-four fixed schedule under the same budget;
- the exact-current-phase constrained oracle;
- the perfect paid-sensing optimum between them.

## Frozen scope and gates

The grid has 120 analytic cells: three normalized steps, two arrival rates,
two persistence values, five stationary rotation fractions, and two budgets.
There are no stochastic outcomes or formal seeds.

The active-sensing research direction survives only if all of the following
hold:

1. LP status, flow, normalization, phase calibration, and call budgets pass;
2. age-256/512 optimal costs differ by at most \(10^{-6}\);
3. exact-phase cost is no larger than perfect-sensing cost, which is no larger
   than the best fixed cost, in every cell;
4. mean upper-bound gain in separated dynamic cells is at least 0.03;
5. at least 75% of those cells have upper-bound gain at least 0.02;
6. low-persistence and low-budget means are at least 0.025 and 0.015;
7. median capture of exact-phase headroom is at least 0.60;
8. stationary potential uses no harmful optimism.

Failure is conclusive for the current action-dependent sensing interface:
because this is an optimistic upper bound, a noisier implementable controller
cannot repair a missing margin.  Passing would only show that a better
controller may exist; it would not validate an algorithm, authorize formal
evidence, or authorize GPU/HPC4 work.
