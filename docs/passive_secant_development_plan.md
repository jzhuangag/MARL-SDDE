# LCO-P0 passive clocked-secant development plan

## Status

LCO-P0 is the first development run for the passive-sensing interface.  It
uses eight new development seeds 84,001--84,008.  It is neither confirmatory
nor formal, and its seeds are permanently excluded from later confirmation.

Unlike LCO-S0 and LCO-V0, the controller never purchases an optimistic update
to obtain information.  Before each asynchronous action it observes the
mandatory current pseudo-gradient.  Together with the preceding mandatory
gradient and parameter state, this gives a passive secant.  The resulting
belief update is available for the current action because neither gradient is
a counterfactual optimistic oracle.

## Frozen comparison

Every path contains:

- the passive-secant Lyapunov controller;
- the exact-current-phase Lyapunov controller as a ceiling;
- never optimism;
- every period-four fixed mask feasible under the same budget.

The strong comparator is the cellwise best fixed mask over the development
seeds.  Optimistic calls are fully charged and share the same hard allowance
for every adaptive or fixed method.  Mandatory gradient observations are part
of every asynchronous update and add no policy-specific oracle call.

The grid has 1,920 paths and 1,966,080 events: three normalized steps, two
arrival rates, two persistence values, five rotation fractions, two budgets,
two mandatory-gradient perturbation levels, and eight seeds.  Dynamics remain
noiseless to isolate passive geometry identification; only the observed
mandatory gradients are perturbed.

## Frozen survival gates

All gates are mandatory:

1. every rate is finite and every optimism call respects its allowance;
2. at each noise level mean separated-dynamic gain over strong fixed is at
   least 0.03;
3. at each noise level median exact-phase gain capture is at least 0.60;
4. at least 75% of separated dynamic cells improve at each noise level;
5. each arrival group has mean gain at least 0.005;
6. low-persistence and low-budget means are at least 0.02 and 0.01;
7. phase-classification accuracy is at least 0.75 at each noise level;
8. informative passive secants cover at least 99% of eligible events at each
   noise level;
9. stationary potential loss and hard-budget overshoot are zero;
10. the result remains development-only.

Failure stops the passive full-gradient interface.  Passing authorizes only a
persistent-excitation/Markov-noise theorem and a separately frozen CPU
confirmation.  It does not authorize formal evidence, a standard MARL
benchmark, GPU, or HPC4.
