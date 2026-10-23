# PDSG-CONE-001 preregistration

Date: 2026-09-06

Status: frozen before any PDSG-CONE-001 calibration or holdout trajectory is
generated.  This experiment uses CPU only and does not evaluate learning
return.

## Research question

Can a launch-state position--velocity predictor, calibrated by split conformal
residuals at fixed stochastic policies, provide a nonvacuous state-dependent
policy-influence tube on standard 20-agent Pistonball while honestly revealing
the horizon-induced sparse-to-dense phase?

## Frozen populations

- Three fixed heterogeneous policy parameter seeds: 913, 1201, and 1777.
- Three public launch-state burn-ins: 5, 15, and 25 joint cycles.
- Three rollout horizons: 4, 6, and 8 joint cycles.
- Twenty continuous-action piston policies with Gaussian pre-`tanh` noise of
  standard deviation 0.3.
- For each of 27 strata, 128 calibration trajectories and 256 disjoint holdout
  trajectories.  Seed numbers are reused across cells only as common random
  numbers; cells are not called independent.

The total frozen workload is 10,368 trajectories.  The experiment reads state
paths only.  Reward, controller return, learning curves, and previous formal
data are outside the analysis population.

## Frozen certificate

The residual is the maximum deviation between the realized ball path and the
launch-time constant-velocity center line, measured in piston widths.  The
split-conformal rank is

\[
\left\lceil(128+1)(1-0.1)\right\rceil=117.
\]

The interaction tube expands the center line by the conformal radius plus two
piston widths for ball/contact geometry.  Every piston in the tube is joined
to every other tube piston by directed policy-cache edges.  The complete-graph
denominator is `20*19=380`.

For a declared per-state joint policy-mean shift of 0.01, policy standard
deviation 0.3, and horizon at most 8, the shifted escape bound is

\[
0.1+\frac{\sqrt 8(0.01)}{2(0.3)}\approx0.14714<0.15.
\]

## Frozen gates

The machine-readable definitions in
`docs/pdsg_cone001_manifest_20260906.json` are authoritative.  C1--C11 are
mandatory.  C12 is run only if they all pass.  In particular, the experiment
must show empirical coverage, short-horizon communication sparsity, horizon
densification, and state-dependent support movement.  A conformal guarantee
without nonvacuous edge savings is not a pass.

Any failed mandatory gate permanently records PDSG-CONE-001 as a failure and
stops the next controller stage.  No threshold, seed, horizon, launch state,
or policy profile may be changed after output inspection under this identifier.

## Interpretation if passed

Passing confirms only the causal-cone certificate and its spatiotemporal phase.
It authorizes design of a separate CPU synthetic Lyapunov-controller audit.  It
does not prove better return, authorize formal controller seeds, or authorize
GPU/HPC4 training.
