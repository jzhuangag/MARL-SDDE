# LCO-S0 dual-use sensor development plan

## Status and purpose

LCO-S0 is architecture development, not a preregistered pilot or formal
experiment.  Its only question is whether a causal hidden-geometry controller
using fully charged dual-use fingerprints retains enough of LCO-H1's
exact-phase headroom to justify a later theorem and independent confirmation.
The eight development seeds are permanently excluded from confirmatory use.

## Frozen causal timing

At event \(k\), the controller first predicts its rotation posterior from the
previous posterior and the public two-state transition law.  It then decides
whether to buy the fresh oracle using only that predicted posterior, resource
debt, and the public forced-probe schedule.  If it buys the oracle, the
fingerprint from the current and lookahead gradients updates the posterior only
for event \(k+1\).  Thus the current action never uses its own counterfactual
observation.

Every forced probe is also the actual optimistic update and is charged against
the same optimism allowance.  There is no separate free sensor.  When the
registered stationary rotation fraction is exactly zero or one, the initial
belief is degenerate and no forced probe is needed to identify a known
stationary control.

## Development model

The state dynamics remain the current-oracle, zero-delay two-dimensional
linear game from LCO-H1.  Fingerprint observations alone receive independent
Gaussian perturbations on both paid gradients; dynamics stay noiseless so this
stage isolates sensing and scheduling rather than stochastic-gradient
convergence.  The Gaussian score likelihood is a declared development model,
not a proved confidence sequence.

The grid contains:

- 8 new development seeds;
- normalized steps 0.2, 0.5, and 0.8;
- first-agent arrival probabilities 0.1 and 0.5;
- phase persistence 0.8 and 0.95;
- stationary rotation fractions 0, 0.25, 0.5, 0.75, and 1;
- optimism budgets 0.25 and 0.5;
- fingerprint noise standard deviations 0 and 0.05;
- fully charged probe periods 4, 8, and 16.

The strong fixed comparator is selected cellwise from every period-four mask
whose call fraction does not exceed the same budget.  The exact-phase
Lyapunov controller is retained only as a sensor ceiling.  Common phase,
arrival, initialization, and per-event standard-normal noise streams are used
across probe periods.

The frozen scientific row order is evaluated with four local CPU workers using
ordered `executor.map`; worker completion order cannot change serialization.
The run command is

`python -m experiments.clocked_async_mpg.run_dual_use_sensor_development run --config docs/dual_use_sensor_development_config.json --output-dir experiments/clocked_async_mpg/results/lco_s0_dual_use_sensor_development --workers 4`.

## Target population and development selection

The separated-dynamic definition is unchanged from LCO-H1:

\[
0<f_R<1,
\qquad
\bar u/f_R\ge p_{\min}(s)+0.05,
\qquad
\bar u\le p_{\min}(s)-0.05.
\]

For each probe period and noise level, report mean log-rate gain over the strong
fixed schedule, improved-cell fraction, contraction fraction, and median gain
capture relative to the exact-phase controller.  The development choice first
maximizes the minimum mean gain across the two noise levels and breaks ties in
favor of the longer probe period.

The architecture survives only if the selected period satisfies all frozen
development gates in the JSON configuration.  Failure stops this hidden-filter
design.  Passing authorizes only an outcome-free confirmation design after the
statistical coverage gap is addressed; it does not authorize formal seeds,
standard MARL, GPU, or HPC4.

## Known limitations before execution

- The true phase-transition law and stationary fraction are supplied.
- The observation likelihood is modeled, not certified under Markov noise.
- The exact stationary controls are identifiable from a degenerate prior.
- State normalization preserves directional excitation and is not a claim
  about an unnormalized nonlinear policy trajectory.
- Best fixed-mask selection uses the same development seeds and is descriptive.
