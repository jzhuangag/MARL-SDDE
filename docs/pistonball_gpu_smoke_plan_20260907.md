## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: compute preflight
- Origin Date: 2026-09-07
- Verification Status: OUTCOME-FREE DEVELOPMENT PLAN
- Version Label: pistonball_gpu_smoke_v1

# Pistonball policy-freshness GPU smoke

This job is a systems smoke, not a scientific pilot.  It uses development seed
78001, one A30, 20 distinct actors, 64 launches, horizon four, randomized
Pistonball initialization, and one final evaluation episode.  No return gate
is defined or permitted.

The only pass conditions are CUDA visibility, finite parameters/returns,
prefix-feasible optional policy bytes, 5,120 charged actor transitions, 57
post-warmup owner-gradient packets received after terminal drain, and zero
packets left in flight.  A failure stops longer jobs for debugging.

The smoke writes only below
`/scratch/jzhuangag/causal-policy-freshness-icml2027`.  It must not write to or
clean `/project`, `/home`, the older `/scratch/jzhuangag/MARL-SDDE`, or any
other active experiment.  Existing L20 jobs are outside its scope.

If the smoke passes, a separate development matrix may estimate learning and
throughput.  New pilot seeds, immutable performance gates, and a fresh commit
are required before any result is called pilot or formal evidence.
