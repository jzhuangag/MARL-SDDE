## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: development experiment plan
- Origin Date: 2026-09-07
- Verification Status: OUTCOME-FREE DEVELOPMENT DESIGN
- Version Label: pistonball_backbone_qualification_v1

# Pistonball backbone learning qualification

The deterministic controller matrix cannot support an efficacy pilot because
all methods tie at the terminal decentralized return and the signed-only term
is inactive.  Before changing the graph objective, this scan tests the more
basic question of whether the current distinct-actor centralized-critic
backbone can learn under an intentionally fully fresh local policy cache.

Five configurations use development-only seed 79003, 327,680 actor
transitions, deterministic CUDA, randomized Pistonball, zero receipt delay,
and identical evaluation seeds.  Four `complete_burst` configurations use a
cone radius of 20 and budget rate 19, allowing every stale teammate in the
20-agent game to be refreshed before a rollout.  They scan actor receipt steps
`0.001`, `0.003`, and `0.01`; one repeats `0.003` with critic step `0.001`.
The fifth configuration uses no refresh with actor step `0.003` and critic step
`0.001` as a diagnostic boundary.

This is not a communication comparison: the fully fresh configurations are a
learner upper-bound diagnostic and their payload is charged but deliberately
unconstrained.  The scan records critic loss, packet-gradient norm, per-actor
parameter drift, returns, exact transitions, packets, payload bytes, runtime,
and memory.

The outcome-free decision threshold is a terminal decentralized-return
increase of at least `2.0` over the common initialization for at least one
fully fresh configuration, together with nonzero finite gradient and actor
drift diagnostics.  Passing retains this actor--critic family for later
controller development.  Failure stops it and redirects implementation to a
strong PPO/MAPPO backbone; it does not trigger further DDPG step-size scans.
No result from this scan is pilot or formal evidence, and seed 79003 is forever
excluded from those populations.
