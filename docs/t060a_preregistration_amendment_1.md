# T-060A preregistration Amendment 1: uint32 seed mapping

## Timing and reason

The first post-preregistration run attempt stopped before the first reference
trajectory was generated.  `MinAtar.Environment.seed` delegates to NumPy's
legacy `RandomState`, which rejects the registered 12-digit provenance seeds
because they exceed `2**32-1`.  The exception occurred before reset, feature
encoding, reference moments, endpoints, output-directory creation, or any
scientific comparison.  Therefore no T-060A outcome existed when this
amendment was written.

## Prospective correction

Every registered environment and policy provenance seed is mapped to the RNG
domain by the deterministic public rule

`rng_seed = provenance_seed mod 2**32`.

The original provenance IDs remain unchanged in the preregistration and
result records.  The mapping applies uniformly to reference, common, and
private streams.  A new test verifies domain validity and exact stream
reproduction for a registered-size seed.

## Unchanged scientific design

Tasks, encoder architecture and seeds, 32 pilot seeds, selection/validation
split, q/rho/overhead/delay grid, reference horizons, learning horizons,
message and environment budgets, endpoints, strong-baseline rule, held-out
oracle rule, V1--V9 thresholds, reproduction requirement, CPU scope, and stop
rule are unchanged.  This amendment does not authorize a controller, formal
seed, GPU, HPC4, or `/project` write.
