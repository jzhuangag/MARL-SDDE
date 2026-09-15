# Preregistration: equal-cost ideal Lyapunov drift sketch

Date: 2026-09-05.

Status: **frozen before outcome generation**.  This audit is an optimistic
necessary condition, not an efficacy experiment and not paper evidence.

## Question and isolation from prior failures

The exact-moment coupled actor--critic oracle has useful headroom, while its
per-packet simultaneous high-probability certificate is provably impractical.
This audit asks the remaining kill question: can a low-dimensional noisy drift
sketch retain enough of that headroom after paying for its own observations?

The audit reuses the already frozen analytic scenario population, which is
development evidence.  It introduces sixteen frozen sketch-noise seeds that
were not used by the exact oracle scan.  No T-083A formal data, public RL
trajectory, return, checkpoint or GPU result enters the runner.

## Optimistic but fully charged information model

Each event has a budget of 256 independent trajectories.  `S1=13` trajectories
estimate the two linear drift coefficients, `S2=13` estimate the three
symmetric quadratic coefficients, and `U=230` are reserved for the applied
update.  The action is a function of `S1`, `S2`, the current moment state and
the version schedule before the `U` innovation is integrated out.

The sensor is deliberately privileged: it observes independent draws of the
latent Gaussian error state of the exact quadratic system.  For an event with
actor and critic response matrices `A,B` and Lyapunov metric `P`, it observes

\[
 \widehat h={1\over |S_1|}\sum_{x\in S_1}
 2\binom{x^\top A^\top Px}{x^\top B^\top Px},
\]

and

\[
 \widehat Q={2\over |S_2|}\sum_{x\in S_2}
 \begin{bmatrix}Ax&Bx\end{bmatrix}^{\!\top}
 P\begin{bmatrix}Ax&Bx\end{bmatrix}+Q_{\rm innovation}.
\]

These are unbiased for the five exact drift coefficients, but the latent
state is not observable in an executable MARL algorithm.  Granting it makes
this an upper-bound test: passing does not establish an estimator; failure is
decisive against any noisier estimator using the same budget.

The equal-cost baselines use all 256 trajectories for their update.  The
sketch update uses only 230, so actor and critic innovation variances are
multiplied by exactly `256/230`.  Sensor trajectories never become update
gradients.  This is the explicit full-cost abstraction of the sealed
`S1/S2/U` split.

## Comparators and metrics

The primary metric is normalized Lyapunov-risk AUC over the same 64
asynchronous completion events.  Comparators are:

1. the per-scenario continuously refined best fixed `(alpha,beta)` using the
   full update batch;
2. the privileged exact online diagonal drift rule using the full batch;
3. the privileged exact coupled rule using the full batch, used only to
   measure recoverable headroom.

For each seeded sketch cell, coefficient decision regret is the accumulated
positive excess of its true conditional drift over the exact split-cost
coupled action, divided by accumulated exact split-cost oracle descent.

## Frozen gates and stopping rule

The machine-readable thresholds and seeds are in
`ideal_drift_sketch_equal_cost_gates.json`.  In particular, the sketch must
improve AUC by at least 10% over the best fixed pair, at least 3% over the
privileged online diagonal rule, beat that rule in at least 60% of cells,
recover at least 60% of the exact coupled-versus-diagonal headroom, and keep
the regret fraction at most 25% in at least 70% of cells.  Target-motion-zero
controls must reduce to the sampled diagonal sketch.

Failure of I4, I5, I6 or I7 permanently stops the coupled-timescale sketch as
the ICML mainline.  Gates, seeds, split sizes, scenario population and metrics
must not be altered after this commit.  A pass authorizes only derivation of a
genuinely observable Markov coefficient estimator.  It does not authorize
formal seeds, GPU/HPC4 work or a standard MARL benchmark.
