# T-063B prospective collision-gate correction

## Status

T-063B is a new prospective validation design.  It is not a reanalysis of
T-063A and does not reclassify the T-063A formal failure.  No T-063B scientific
trajectory is authorized until the static audit, tests, hashes, and a separate
commit are complete.

## Unchanged scientific claim

T-063B keeps the T-063A tasks (Asterix, Breakout, Seaquest), fixed uniform
policy, MinAtar 1.0.15 environment, frozen nonlinear encoder, affine TD head,
correlation grid, delay grid, dual budgets, reward-free fingerprint controller,
strong fixed-q comparator, efficacy gates, bootstrap, and 512-seed power
allocation.  It uses only new seeds `202608057201` through `202608057712`.

## Corrected collision gate

T-063A used a maximum match rate over 1,536 blocks.  T-063B instead sums the
96 probe match counts over all rho=0 seed-task blocks and computes the exact
one-sided Clopper--Pearson upper confidence bound at alpha `0.05`.  The gate
passes only when that upper bound is at most the frozen independent-path
probability bound `0.0007716049382716049`.  The blockwise maximum remains in
the output as descriptive information only.

This correction is locked before T-063B seeds are run and is motivated by the
post-result multiplicity audit in `docs/t063a_p10_boundary_audit.md`.  It does
not change any efficacy threshold or use T-063A outcomes as evidence.

## Stop rule

Any T-063B coverage, finite-cost, efficacy, collision, replay, byte-exact
reproduction, or test gate failure stops the new claim.  The result must be
reported alongside the unchanged T-063A failure; no threshold, seed, task,
comparator, or analysis rule may be altered after execution begins.
