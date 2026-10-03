# LCO-P0 passive clocked-secant development validation

## Decision

The raw passive-secant controller fails 8 of 12 frozen gates and the geometry
sensing mainline is stopped.  It is not eligible for confirmation, formal
evidence, or a standard MARL/GPU benchmark.  Thresholds, seeds, noise levels,
and target cells were not changed after execution.

The run itself completed correctly: 1,920 unique paths, 1,966,080 events, no
nonfinite rate, no budget violation, and exact semantic reanalysis from the
stored rows.

## Frozen outcome

- execution commit: `c83177b`;
- configuration SHA-256:
  `fb467ec266003718fc6885a8fa909ffdce83c940e5911bf1cb8c54b09f32bad0`;
- result SHA-256:
  `24f10369f2a5e54b90466700559060c18d889e4cb58140097fb69673005132b2`;
- development seeds: 84,001--84,008, never eligible for confirmation.

| Metric | Noise 0 | Noise 0.05 |
|---|---:|---:|
| Mean dynamic gain over strong fixed | 0.040163 | -0.037551 |
| Improved dynamic-cell fraction | 0.84375 | 0.40625 |
| Median exact-phase gain capture | 0.715531 | -0.409900 |
| Mean phase accuracy | 0.970321 | 0.703486 |
| Mean informative-secant fraction | 0.758801 | 0.791052 |

With exact mandatory gradients, passive secants cross the performance,
coverage-of-cells, accuracy, and exact-capture objectives.  This confirms the
deterministic identifiability mechanism.  With additive gradient noise 0.05,
however, performance reverses sign and stationary-potential loss reaches
0.224933.  Low-budget mean gain is -0.006912.

## Why this is a stochastic-optimization failure

The raw statistic divides a gradient difference by displacement energy.  If

\[
\widehat{\Delta g}=A\Delta x+\Delta\xi,
\]

then its noise term scales as

\[
\frac{\langle\Delta x,\Delta\xi\rangle}{\|\Delta x\|^2}
\quad\text{and}\quad
\frac{\|\Delta\xi\|}{\|\Delta x\|}.
\]

Small asynchronous coordinate moves therefore amplify ordinary stochastic
gradient noise.  The observed failure is not a cosmetic likelihood mismatch:
the sensor becomes least reliable exactly when stability calls are scarce,
causing harmful optimism in potential phases.  Screening, smoothing, or
window tuning after this outcome would create another unregistered estimator
and is not authorized by the frozen stop rule.

## Program-level consequence

The complete geometry-adaptive optimism sequence now contains:

- positive exact-phase headroom (LCO-H1);
- failure of fixed paid probing (LCO-S0);
- failure of one-step dual control (LCO-V0);
- a negative perfect-paid-sensing capture gate (LCO-U0);
- deterministic success but stochastic failure of passive secants (LCO-P0).

Together these results are sufficient to stop the line rather than search for
a favorable standard benchmark.  The next ICML candidate must use quantities
that are directly observed and well-conditioned under Markov gradient noise.
The recommended pivot is a two-clock asynchronous learning problem based on
actor arrival clocks, packet age, Markov transition stride, and agent-coverage
debt.  These are observable without differencing stochastic gradients.

No clean reproduction was run because all survival gates were mandatory and
the architecture failed.  The run was local CPU only; no GPU, HPC4, or remote
storage was used.
