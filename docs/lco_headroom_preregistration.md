# LCO-H1 preregistration: current-oracle phase headroom

## Frozen purpose

LCO-H1 tests whether phase-dependent optimism has material endogenous value in
the exact current-oracle asynchronous coordinate-game subclass.  It is a CPU
theorem/headroom scan, not a Markov policy-gradient experiment, formal
evidence, or GPU authorization.

- configuration: `docs/lco_headroom_config.json`;
- configuration SHA-256:
  `58CACEEEA755D8A1057073EEAE0CCA9284ABC0F4F8E139C695C7D834EB54F6B8`;
- 8,640 stochastic paths and 8,847,360 coordinate events;
- 32 fresh analytic seeds beginning at 81001;
- 270 seed-aggregated cells;
- local CPU only; no HARL, HPC4, GPU, or prior formal data.

The grid crosses normalized steps 0.2/0.5/0.8, first-agent arrival
probabilities 0.1/0.3/0.5, phase persistence 0.8/0.95, stationary rotation
fractions 0/0.25/0.5/0.75/1, and optimism budgets 0.25/0.5/0.75.  The two-state
phase chain has the registered stationary rotation fraction and second
eigenvalue equal to the persistence parameter.

## Controller and comparators

The controller observes the exact linear phase certificate, uses the
clock-balanced metric, and applies the frozen `V=sqrt(horizon)` log-drift queue
rule.  This is an optimistic sensor ceiling: a later stochastic MARL algorithm
must estimate the phase without outcome leakage.

Every path shares common random phases, arrivals, and initialization across:

- LCO;
- phase-aware same-count oracle;
- never optimism;
- always optimism, with its full doubled-oracle cost reported;
- every phase-independent length-four fixed mask using no more than the same
  optimism budget.

The strong fixed comparator is the mask with lowest mean log-energy rate over
all 32 analytic seeds in its cell.  It is selected only for this feasibility
audit and cannot later be called a formal baseline selected independently of
outcomes.

## Separated dynamic population

A mixed-phase cell is separated when

\[
\bar u/f_{\rm rot}\ge p_{\min}(s)+0.05,
\qquad
\bar u\le p_{\min}(s)-0.05.
\]

Thus concentrating the budget in rotational phases is theoretically
sufficient, whereas a phase-independent use rate is below the exact
rotational boundary.  This population is defined entirely from the theorem and
frozen config, before outcomes.

## Mandatory gates

All ten gates must pass:

1. finite results and exact resource accounting;
2. mean controller log-energy-rate gain over the best fixed mask at least
   0.005 in separated cells;
3. strict improvement in at least 80% of separated cells;
4. contraction in at least 80% of separated cells;
5. median phase-oracle gain capture at least 60%;
6. zero optimism calls in stationary potential controls;
7. controller and never log-energy rates agree within `1e-12` there;
8. mean separated-cell gain at least 0.002 separately for every registered
   arrival probability;
9. zero hard-budget overshoot;
10. any failed gate stops Markov-noise development, standard MARL, formal, and
    GPU escalation.

No configuration, gate, seed, comparator, or separated-population rule may be
changed after the scan.  Even a full pass authorizes only the next CPU audit:
constructing predictable stochastic symmetric/skew certificates and adding
Markov critic noise.
