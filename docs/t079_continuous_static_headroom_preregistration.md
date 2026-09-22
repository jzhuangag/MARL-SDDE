# T-079 continuous-static collaboration headroom preregistration

## Purpose

T-070A established dynamic collaboration value against all 2,401 registered
discrete static graphs.  The continuous-simplex controller introduced later is
strictly richer than that catalogue, so the discrete comparator is not strong
enough for the proposed online-collaboration mainline.  T-079 asks whether the
headroom survives after strengthening the static baseline to one arbitrary
row-stochastic matrix reused at every decision block.

T-079 is an exact-moment, outcome-aware comparator audit.  It creates no sampled
trajectory and is not pilot or formal evidence.  Its sole purpose is to decide
whether a new-seed observable-controller experiment is scientifically justified.

## Frozen comparison

For each of the unchanged 432 T-070A cells, the static baseline minimizes exact
24-block cumulative personalized mean-square risk over a continuous 4-by-4
row-stochastic matrix.  Because repeated collaboration makes the full-horizon
objective nonconvex, the implementation makes no global-optimality claim.  It
uses ten deterministic SLSQP starts, including the frozen best discrete graph,
so the resulting comparator is guaranteed computationally to be no weaker than
that discrete baseline and is explicitly reported as the strongest solution
found.

The dynamic ceiling minimizes each recipient's exact next-block quadratic risk
over its entire probability simplex.  All nonempty supports are enumerated, so
this row-wise subproblem is globally solved up to floating-point tolerance.
The dynamic policy uses all 240 transitions for learning, consumes no extra
probe transition, charges one fingerprint-message unit at each decision, and
charges each nonlocal mixing decision.

## Primary decision

The primary comparison is the geometric cumulative-risk ratio of the safe
dynamic continuous oracle to the strengthened continuous static graph on all
288 nonstationary cells.  The frozen headroom gates require at least 5% aggregate
improvement, strict improvement in at least 60% of cells, at least 3% in both
schedule families, and nonnegative improvement for every delay.

Any mandatory failure stops a new-seed controller pilot.  No threshold, cell,
optimizer start, or source artifact may be changed after the independent
preregistration commit.  Full gates and provenance are frozen in
`t079_continuous_static_headroom_preregistration.json`.

## Authorization

After an independent preregistration commit, the unchanged audit may run on the
local CPU.  T-079 does not authorize sampled pilot seeds, formal evidence,
nonlinear benchmarks, GPU, HPC4, or `/project` writes.
