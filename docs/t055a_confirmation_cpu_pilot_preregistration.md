# T-055A independent CPU confirmation pilot preregistration

## Scientific question

T-053A retained an aggregate 10.96% gain on three exact Gymnasium kernels but
failed its per-task gate on CliffWalking with only eight paired master seeds.
T-054 showed that its seed-cluster interval was too wide to distinguish the
T-052A prediction from no gain. T-055A asks the same frozen scientific
question with 64 entirely new master seeds.

This is a new confirmation pilot, not a rerun, extension, or amended analysis
of T-053A. The T-053A 11/12 failure remains permanent. No T-053A seed becomes
a T-055A seed. T-053A outcome rows determine only the prospective sample size
through the committed T-054 power audit; they do not alter the algorithm,
tasks, cells, comparators, thresholds, or estimator.

## Frozen design

The three public fixed-policy Gymnasium kernels, delays {0,8}, overheads
{8,32}, seven correlations, q catalogue {1,4,16}, contraction horizons,
two-agent fingerprint probe, fully charged budgets, exact trajectory-switch
coupling, and all P1--P12 thresholds are unchanged from T-053A. The design has
84 cells, 64 new seeds, 5,376 paired endpoints, and 91,392 sequentially
generated long learning paths. Full trajectories are not stored.

The same common-plus-16-private trajectory bank is shared through common
random numbers by the controller and fixed-q comparators within an endpoint.
The unit of independent replication is the master seed, not an endpoint row.

## Frozen stop rule

Every mandatory gate is conjunctive. In particular, the aggregate ratio must
be at most 0.95, every task and delay ratio at most 0.97, at least 60% of
oracle-active cells must improve, inactive cells must remain within 1.05, and
the controller must remain within 1.20 of the true-rho oracle. Any failure
forbids formal execution and cannot be repaired by changing seeds, cells, or
thresholds.

Passing authorizes only a separate formal preregistration. T-055A itself is
not formal paper evidence and does not authorize a nonlinear benchmark, GPU,
HPC4, or use of `/project`. The expected local CPU time is about 6--8 minutes.

The frozen CLI entry point is
`python -m experiments.dependence_delay_linear.run_t055a_confirmation_cpu_pilot`.
Both `validate` and `estimate` must pass before the one permitted `run`.
