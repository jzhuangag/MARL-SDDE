# Clocked asynchronous MPG intermediate validation

Date: 2026-09-01

## Outcome

The exact-gradient event-time Lyapunov--Krasovskii lemma and its conditionally
centered packet-noise corollary pass their bounded algebraic validation.  This
closes one intermediate proof interface only.  It does not authorize a
scientific trajectory, a CPU efficacy pilot, a standard MARL benchmark or a
GPU job.

## What was checked

- interaction-weighted delay-history coefficients agree with their analytic
  definition;
- the closed-form maximum constant step saturates the blockwise stability
  condition and reduces to the block-smooth restriction at zero delay;
- the exact-gradient one-event drift upper bound holds on random coupled
  positive-semidefinite quadratics;
- the centered-noise extension holds under exact two-point innovation
  enumeration and reduces numerically to the exact-gradient calculation at
  zero variance;
- non-finite noise scales are rejected;
- the pre-application filtration explicitly excludes the current arriving
  packet innovation.

## Commands and results

```text
conda run -n ust2 python -m pytest experiments/clocked_async_mpg -q
15 passed in 0.20s

.venv/Scripts/python.exe -m pytest -q experiments
939 passed, 7 skipped in 124.78s
```

The repository `.venv` reported NumPy 2.2.2 and CVXPY 1.6.5.  A preliminary
root-level run in `ust2` was invalid as a full-regression check because it both
lacked CVXPY and recursively collected preserved test snapshots under `tmp/`.
Those collection errors are environment/scope errors, not test failures.  No
package was installed and no snapshot was deleted.

## Proof audit and stop boundary

The stochastic corollary assumes that, conditional on the pre-application
history and predictable packet metadata, the arriving innovation remains
centered.  Marginal unbiasedness is insufficient.  In particular, if faster
completion is correlated with a trajectory statistic or gradient innovation,
first-arrival selection can create nonzero conditional bias.  The current
theorem therefore covers only non-informative completion or an estimator with
a separately proved selection correction.

The result is also limited to scalar smooth blocks, iid agent activation,
bounded event delay, Euclidean unconstrained updates and a variance-bounded
centered estimator.  It does not yet contain a Markov mixing-bias term, critic
tracking, vector policy geometry, a potential-to-Nash conversion or a
wall-clock service model.  Until these interfaces close without reducing to a
generic delayed stochastic-approximation corollary, the candidate fails the
ICML novelty/completeness gate and no efficacy experiment is authorized.
