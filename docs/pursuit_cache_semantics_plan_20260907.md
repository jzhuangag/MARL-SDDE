## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: outcome-free interface qualification plan
- Origin Date: 2026-09-07
- Verification Status: FROZEN STRUCTURAL AUDIT; NO REWARD OUTCOME
- Version Label: pursuit_cache_semantics_qualification_v1

# Pursuit cache and topology semantics qualification

## Scope

This CPU-only audit closes the state-machine interface required by the
two-layer Lyapunov theorem. It uses actual PettingZoo Pursuit state sequences
but never evaluates or aggregates reward. It is not an efficacy experiment and
cannot authorize a GPU run by itself.

The audit uses seeds `92000--92007`, action seed `9981`, eight pursuers, thirty
evaders, forty cycles, observation range seven, and the previously qualified
degree-four state-dependent candidate graph. A small distinct actor is used to
exercise parameter copying and delayed packet application; no actor is trained.

## Frozen checks

1. At least one non-null cache refresh is executed.
2. Every non-null refresh charges the full donor actor payload at launch.
3. On the fixed directed edge universe, the launch refresh decreases cache
   energy by exactly the selected-edge reset.
4. A packet carrying zero receipt weight changes no actor or actor version, but
   its already charged launch-time refresh remains in the cache.
5. Pursuit changes the state-dependent candidate support.
6. The active-edge cache-energy change under support turnover equals the exact
   topology-motion identity with policies and caches held fixed.
7. At least one positive topology-motion jump is observed, demonstrating that
   it cannot be silently discarded.

All seven checks must pass. The primary and an isolated reproduction JSON must
be byte-identical. A failure changes the interface or theorem; it is not
repaired by loosening a check after execution.

## Frozen source hashes

- audit runner: `aae486fa27b1bf76bd0642921d003fd2a8a16f5805c6b32975cd33e0e2a28dfe`;
- packet/cache state machine: `fc5f5250fa30cc8345df02dc43df11816042e9d5808cebbf0bc3f5cc6c98f899`;
- cache/topology algebra: `770d412940584a2383b2c8e6292b52d5f1a6b2699cbce63848628c0cb56a700b`;
- Pursuit local-factor graph: `35db3b56898a35a2246c52e6d1f4c94ac021b6f8620b38a94a35f3ced7838d5a`.

The audit writes below ignored `experiments/policy_dependency_sync/results/`.
No GPU, HPC4, `/project`, or remote storage action is authorized.
