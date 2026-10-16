# SMACv2 moderate-dependence headroom preregistration

## Question and separation from DEV-001

`TSP-MARL-SMACV2-HEADROOM-001` asks whether a standard 10-versus-10 CTDE task
contains enough *performance* value for correlation-adaptive rollout
participation.  It does not relabel the stopped `TSP-MARL-SMACV2-DEV-001`:
DEV-001 assumed a near-replication shared regime and failed its frozen
coupling qualification.  The outcome-free calibration instead established
two regimes on opposite sides of the controller's pre-existing correlation
boundary: independent residual streams had median correlation `-0.01484`,
whereas common-seed/action streams had median correlation `0.50066`.  All four
frozen participation decisions were correct.  No return or win rate was
observed in that calibration.

The new experiment therefore treats common-seed SMACv2 as an intrinsically
moderate-dependence simulator regime.  Before spending computation on a
controller, it tests the stronger necessary condition that the two fixed
participation levels have a meaningful regime-dependent learning-value gap.

## Frozen design

The task is SMACv2 `terran_10_vs_10` with parameter-shared MAPPO actors and a
centralized critic.  Decentralized evaluation uses deterministic actions.
Fixed `q=1` and fixed `q=8` are crossed with independent and common-seed/action
rollout regimes and two new development seeds, `81101` and `81102`, for eight
runs.  There is no probe and no adaptive controller in this stage.

The physical budgets are 6,400,000 message units and 1,200,000 environment
ticks.  A block of length 200 costs `800+200q` messages and 200 environment
ticks.  Thus `q=1` receives 6,000 updates (1,200,000 actor transitions) and
`q=8` receives 2,666 updates (4,265,600 actor transitions).  Networks have two
128-unit ReLU layers; actor and critic learning rates are `5e-4`; MAPPO uses
five actor and five critic epochs and value normalization.  Evaluation uses
32 episodes in eight threads every 200 learner updates.

The primary endpoint is team-return AUC on a common 21-point fully charged
budget-fraction grid.  Win-rate AUC is secondary because this development
horizon may precede reliable wins on 10-versus-10 combat.

## Frozen gates and stopping rule

All eight runs must be unique, finite, exactly charged, use clean pinned
upstream code, and finish without timeout.  Mean return-AUC for `q=8` must
exceed `q=1` by at least 5% in the independent regime; mean return-AUC for
`q=1` must exceed `q=8` by at least 5% in the common-seed regime.  The mean of
the two regime-wise fixed-q oracles must exceed the best single fixed q under
an equal regime mixture by at least 3%.  Terminal-return directions must agree
with both AUC directions.  A clean analyzer replay must be byte-identical.

Any failure stops the SMACv2 controller stage without changed thresholds,
seeds, budgets, or task.  Passing authorizes only a new, separately frozen
controller development stage; it is not confirmatory evidence and does not
enter the manuscript automatically.
