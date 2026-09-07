## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: independent CPU confirmation preregistration
- Origin Date: 2026-09-07
- Verification Status: FROZEN DESIGN; DO NOT RUN BEFORE PREREGISTRATION COMMIT
- Version Label: PDSG-FR-001-preregistration-v1

# PDSG-FR-001: Markov forecast-reversal confirmation

## Question

Can an online plug-in Lyapunov controller recover a nontrivial fraction of the
finite-horizon Markov graph oracle's learning value, while respecting a
communication budget, when the current instantaneous factor ranking is
strategically stale for the trajectory that will generate the next delayed
owner update?

This is a confirmation of one mechanism, not a standard MARL result.  The
development seeds `97100--97107` are permanently excluded.  Confirmation uses
32 untouched seeds `98100--98131` and the unchanged 24-cell matrix, ten
policies, 4,096 event horizon, controller, model, and analysis definitions in
`pdsg_fr001_confirmation_config.json`.

## Primary estimand and fixed comparator

The primary population is the 12 cells with cycle probability at least `0.8`.
The estimand is geometric cumulative learning risk
(`average_potential`) of `plugin_joint` divided by `state_myopic`, first within
each cell over 32 common-random-number seeds and then over cells.  Development
selected `state_myopic` in all 24 cells, so confirmation does not select a new
baseline from its outcomes.  All fixed, periodic, random, no-refresh, and exact
dynamic-oracle policies remain in the run for auditing and ablation.

Actual optional messages, total Markov transitions, terminal risk, queue, and
seedwise direction are retained.  Wall-clock is not a scientific outcome.

## Frozen mandatory gates

1. `P1`: exactly `24*32*10=7,680` unique endpoint identities.
2. `P2`: all risk/parameter outputs finite and environment-transition counts
   equal within every scenario/seed.
3. `P3`: no communication-budget violation, using the proved queue slack for
   joint controllers and the hard periodic cap for baselines.
4. `P4`: favorable geometric cumulative-risk ratio at most `0.75`.
5. `P5`: strict favorable improvement in all `12/12` cells.
6. `P6`: favorable seedwise win fraction at least `0.90` over 384 paired cases.
7. `P7`: median favorable exact-oracle headroom recovery at least `0.75`.
8. `P8`: all 12 favorable plug-in points are return--message Pareto dominant
   over at least one state-myopic point with no fewer actual messages at the
   same cycle probability and delay.
9. `P9`: low-cycle minus favorable geometric ratio is at least `0.25`, testing
   the analytic forecast-value phase rather than universal dominance.
10. `P10`: all-cell geometric cumulative-risk ratio at most `0.85`.
11. `P11`: the exact dynamic oracle retains an all-cell ratio at most `0.80`,
    confirming that the environment still contains dynamic headroom.
12. `P12`: an isolated rerun must reproduce `endpoints.csv`, `summary.json`,
    and the scientific fields of `validation.json` byte-for-byte; execution
    timing is excluded and no timing field is written.

Any failed gate is reported without changing seeds, cells, thresholds,
controller constants, or comparator.  Failure stops the standard-MARL bridge.
Passing all gates permits only the next outcome-free Pursuit critic/interface
design.  It does not itself authorize a GPU pilot or a formal paper claim.

## Evidence controls

- The score matrix has zero stationary mean for every fixed factor.
- The analytic two-step rank boundary is fixed at
  `c_star=0.488888...`; the primary population was chosen above this boundary.
- Scores for all three local factors use the same charged trajectory and are
  the tabular counterpart of an `O(Delta)` centralized local-factor critic.
- Zero-weight refreshes send no packet and incur no message.
- Owner updates, optional messages, random delays, queues, and drain packets
  are all included.
- The hard-LCB shield is not the primary controller; its finite-sample
  conservatism is already recorded.  The theorem-facing uncertainty cost is
  the cumulative plug-in oracle term in
  `plugin_lyapunov_oracle_bridge_20260907.md`.

## Execution boundary

CPU only.  No HPC4, GPU, `/project`, or remote storage action is authorized.
The primary may run only after this document, the JSON configuration, runner,
analyzer, tests, and their hashes are committed.  Reproduction uses an
isolated output directory.  Formal seeds and a standard MARL benchmark remain
unassigned.

Frozen preregistration hashes before execution:

- configuration JSON:
  `f25c7ba61037650572c3716ad656f57d5906f9b830caf8b8884f1367acd08238`;
- confirmation runner/analyzer:
  `2fc2245842355fab371f6b00990258b1951ec5009e4506128bdf7be223f617e1`.
