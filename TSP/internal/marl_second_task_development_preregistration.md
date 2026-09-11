# TSP-MARL-XTASK-DEV-001 preregistration

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-11
- Verification Status: PREREGISTERED-DESIGN
- Version Label: xtask_development_v1

## Purpose

This outcome-free development experiment asks whether the already confirmed
dependence-selection mechanism transfers from spatial coverage in
`simple_spread_v2` to communication-conditioned cooperative navigation in
`simple_reference_v2`.  It is a task-diversity test, not a search over new
controller formulas.  The probe statistic, candidate participation actions,
budgets, optimizer settings, network, and evaluation protocol are copied from
TSP-MARL-CONF-001.

Passing this development screen authorizes only a separate confirmation
preregistration with unseen seeds.  Development outcomes cannot enter the
manuscript as confirmatory evidence and cannot be used to alter the frozen
TSP-MARL-CONF-001 result.

## Outcome-free compatibility audit

The pinned HARL environment uses PettingZoo 1.22.2 and SuperSuit padding.  A
remote import-only check at the pinned HARL commit reports two agents, padded
observation shapes `(21,)`, padded action spaces `Discrete(50)`, and repeated
centralized state shapes `(42,)`.  Thus the frozen shared-parameter MAPPO
interface is shape-compatible.  This check observed no return, gradient, or
training trajectory.  The subsequent two-cell smoke uses separate seeds
`105991--105994`, eight probe blocks, and tiny budgets solely to verify that
policy sharing, critic-residual collection, shared categorical coupling, and
exact cost accounting execute together.

## Frozen design

- Task: discrete PettingZoo MPE `simple_reference_v2`, MAPPO, CTDE with
  decentralized deterministic evaluation.
- Upstream: clean HARL commit
  `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2` and PettingZoo 1.22.2.
- Methods: unchanged Lyapunov probe-then-commit controller, fixed `q=1`, and
  fixed `q=8`.
- Regimes: marginal-preserving independent and shared rollout innovations.
- Controller: 128 fully charged non-learning `q=8` probe blocks, the existing
  one-sided correlation proxy, and one irreversible choice from `{1,8}`.
- Training: actor/critic rates `0.0005`, two PPO and critic epochs, two
  64-unit tanh layers, episode/rollout length 25.
- Budgets: 5,000,000 messages and 1,000,000 environment ticks; per-block cost
  is `100+25q` messages and 25 environment ticks.
- Evaluation: 32 deterministic episodes every 1,000 learner updates; primary
  statistic is unsmoothed team-return AUC on a 21-point maximum charged
  dual-budget-fraction grid.
- Development training seeds: `106001--106004`.
- Disjoint probe seeds: `107001--107004`.
- Total: 2 regimes x 4 seeds x 3 methods = 24 runs.

No task return is visible to the controller.  No training or probe seed used in
the earlier development or confirmation studies is reused.

## Frozen development gates

All gates are mandatory:

1. exactly 24 unique finite records, exact dual-budget accounting, and a clean
   pinned upstream checkout;
2. at least 3/4 independent controller runs select `q=8`;
3. at least 3/4 shared controller runs select `q=1`;
4. independent mean controller AUC relative to fixed `q=8` is at least -2%;
5. shared mean controller AUC relative to fixed `q=8` is at least +3%;
6. equal-weight independent/shared mean gain relative to fixed `q=8` is at
   least +1.5%;
7. probe messages are at most 1% of the message budget;
8. scalar controller overhead is at most 5% of training time;
9. the existing marginal-preserving coupling audit passes.

Any failure yields `stop`, preserves the complete development result, and
prohibits replacement development seeds or a confirmatory claim.  A passing
screen permits a new, disjoint eight-seed confirmation preregistration without
changing the controller or the TSP-MARL-CONF-001 evidence.

## Frozen provenance

- Machine-readable design SHA-256:
  `6dc2eae1407d14f8c5523ab45343187ea1680c004562a8193fce23ba1cf16be7`.
- Controller SHA-256:
  `14e1f3941f734a7652726251aacb7f59fb21485b9e1387cf29100722d0c2b9c9`.
- Controller runner SHA-256:
  `b901693f4705de575e01d4068d9803295359e8733f3d498bf1f47d8693b79164`.
- Fixed-action runner SHA-256:
  `f3e872bb48a52ce1cd14f1358076d503ca17aabe15535b023567d8cd8fd93fdf`.
- Development analyzer SHA-256:
  `8138fadf4c68ffbd4a9ffe3c5621cc627cc5f998e79519195a81722c8ccb073b`.
- Coupling audit SHA-256:
  `b39be0d3312585c4a89a7e36ceaf5d72d60b59e4a7d628a215506285708c631d`.

Scientific outputs belong exclusively under
`/scratch/jzhuangag/MARL-SDDE-TSP-MARL-XTASK-DEV-001`.  The smoke uses the
separate root `/scratch/jzhuangag/MARL-SDDE-TSP-MARL-XTASK-SMOKE-001`.  Neither
stage writes to `/project` or `/home`.
