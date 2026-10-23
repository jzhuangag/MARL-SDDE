## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: outcome-free benchmark selection
- Origin Date: 2026-09-08
- Verification Status: PRIMARY AND SECONDARY CANDIDATES SELECTED; HEADROOM NOT YET ESTABLISHED
- Version Label: replacement_benchmark_decision_v1

# Replacement benchmark decision after the Pursuit stop

## Selection question

The standard task must instantiate the paper's causal freshness problem before
any controller outcome is read.  Popularity alone is insufficient.  The
minimum contract is:

1. distinct policy blocks and simultaneous-action training;
2. a public local teammate-dependency interface;
3. an informative learning signal at the packet horizon;
4. configurable agent count and asynchronous service;
5. exact transition and policy-byte accounting;
6. a parameter-shared critic with candidate cost linear in local degree;
7. nontrivial equal-resource oracle value over strong online schedulers.

## Primary candidate: Multiwalker

PettingZoo Multiwalker is selected for the next outcome-free contract.  It is
not yet selected as a positive paper result.

- The current reference task has a Parallel API and configurable `n_walkers`.
- Each walker has a continuous four-dimensional action and a 31-dimensional
  observation containing its body state, local lidar, neighboring walkers, and
  the package.
- The per-step progress reward is the package's horizontal displacement times
  a public scale.  This is much denser than capture-only Pursuit feedback.
- The public neighbor observation creates a chain-local factor universe.  The
  selected directed cache graph can still change at every launch even though
  the physical neighbor universe is fixed.
- `position_noise`, `angle_noise`, fall, and termination events support Markov
  noise and tail-risk evaluation without inventing a new task reward.

The proposed experiment uses five walkers for the primary scaling point, with
three and eight as secondary sizes if the environment remains stable.  CTDE
holds a centralized package/team critic and per-recipient teammate-policy
caches; each actor executes only from its local observation.  Asynchronous
launch/receipt and optional policy synchronization are training-time only.

The local `ust2` environment contains PettingZoo 1.26.1 but not its optional
Box2D dependency.  No scientific run is authorized until an isolated,
reproducible environment is created and a reset/step/cache contract passes.

## Secondary candidate: Knights--Archers--Zombies

KAZ is selected as the heterogeneous-agent stress task, conditional on the
primary contract.  The installed PettingZoo v10 task already resets and steps
locally with vector/typemask observations.

- Knights and archers have distinct role dynamics while sharing a six-action
  discrete interface.
- The vector observation exposes typed agent, weapon, projectile, and zombie
  states; a distance/line-of-effect graph can therefore be declared before
  outcome inspection.
- Dynamic deaths, arrows, swords, and zombie arrivals produce genuine support
  turnover.

Its kill reward is sparser and role-local, so the centralized learner must use
the predeclared team objective rather than silently changing the environment
reward after inspection.  KAZ cannot replace Multiwalker merely because it is
locally available; it must separately pass an information/headroom gate.

## Rejected or reduced candidates

- **Pursuit:** permanently stopped as principal efficacy evidence after the
  final raw-observation critic failed all four estimator-class gates.  Retained
  for structural dynamic-topology and cache semantics.
- **Pistonball:** retains the earlier dense/high-degree degeneration role.  Its
  shaped reward is attractive, but the registered H-step cone required 9--18
  reverse evaluations.  A new factor critic is not enough to erase the cone's
  physical degree.
- **Waterworld:** its range sensors, configurable cooperation, encounter
  shaping, and local/global reward mixture are conceptually attractive, but
  it was removed from PettingZoo 1.26 and is absent from the project runtime.
  Pinning an obsolete environment as the main 2027 result would create an
  avoidable maintenance/reproducibility liability.
- **KAZ as the sole primary:** rejected because the untrained kill signal can
  be sparse and its default individual credit differs from the shared team
  objective required by the theorem.

## Frozen order of work

1. Build an isolated Multiwalker environment lock; perform reset, deterministic
   replay, per-recipient cache, delayed receipt, and exact byte-accounting
   tests without reading a learning comparison.
2. Define the fixed chain edge universe and continuous-policy profile factor.
   Derive the deterministic-policy/Gaussian-policy directional alignment and
   its `O(Delta)` Lyapunov insertion.
3. Before fitting the controller, run an equal-resource oracle ceiling against
   no refresh, all neighbors, age, mismatch, periodic, random, best fixed edge,
   and strong online myopic policies.  The primary threshold is at least 10%
   learning-value improvement and broad directional cells.
4. Only if the ceiling passes, train a selected-packet critic with disjoint
   representation/head/calibration/test data and a nonvacuous error allowance.
5. Only after those CPU gates pass may a standard-task learning pilot be
   preregistered.  GPU is then likely useful, but not yet authorized.

This order preserves one story: the learning Lyapunov function prices which
teammate policy version should enter the next asynchronous trajectory and how
strongly its returned owner packet should be applied.  The benchmark supplies
the endogenous physical value; it is not tuned to manufacture a positive
controller result.

## Public source record

- PettingZoo Multiwalker documentation:
  <https://pettingzoo.farama.org/main/environments/sisl/multiwalker/>
- PettingZoo KAZ documentation:
  <https://pettingzoo.farama.org/main/environments/butterfly/knights_archers_zombies/>
- PettingZoo Pistonball documentation:
  <https://pettingzoo.farama.org/environments/butterfly/pistonball/>
- PettingZoo 1.26 release record documenting Waterworld removal:
  <https://github.com/Farama-Foundation/PettingZoo/releases>

These are task/interface sources, not evidence that the proposed method
improves learning.
