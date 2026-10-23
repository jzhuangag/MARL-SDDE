# TSP-V2 online Lyapunov MARL development preregistration

## Status

This commit freezes `TSP-V2-MARL-ONLINE-DEV-001`.  It is a development
experiment, not confirmation and not manuscript evidence.  Its only allowed
outcome is a decision on whether a separately preregistered, fresh-seed
confirmation is justified.

The executable interface and conditional finite-time theorem were fixed at
implementation commit `654b7b6`.  G0-B traces were used only as earlier
interface/design information.  No final evaluation return from this
experiment existed when the settings and gates below were frozen.

Frozen SHA-256 values are:

- configuration: `BFCEC379B2C1D3E91A0A98FC04B031EA565B86819863A4C840DD169FC3C61C94`;
- runner: `5979B34AC234BE73D7FE15956BD5F40DCE9149E55CBD0777E33462118AF07259`;
- analyzer: `CDBB757D3A03BEE6FBC4CCAB3F29E5A6096E56D3CDCA51B91138F17140EF70A3`;
- controller: `F61A5BC1E1E92D366D8396C223003EAA9E3152FA4307BADB970381326CE9A2FD`;
- Slurm script: `CEC7B92785451B242D30E1EF0B53A31111E506672A7F3FCD4FBC35079C3EAE5F`.

## Scientific question

Under the same message and environment-tick budgets, can an online controller
that observes only fully charged, disjoint validation progress learn when
larger parallel participation is useful and achieve a better return curve than
the strongest fixed choice from `q={1,2,4,8}`?

The controller is not the earlier correlation-only rule.  It learns the
realized finite-horizon risk reduction of each participation action and prices
its two known resource costs through Lyapunov queues.  This directly tests the
TSP-V2 mechanism on one discrete PettingZoo MPE task and one continuous
MaMuJoCo task.

## Frozen design

- CTDE learner: HARL MAPPO with decentralized deterministic evaluation.
- Tasks: PettingZoo MPE `simple_spread_v2`; MaMuJoCo
  `HalfCheetah-v2/2x3`.
- Coupling regimes: independent and shared rollout randomness, without
  changing any single-worker environment marginal.
- Methods: online Lyapunov controller and fixed `q=1,2,4,8`.
- Development seeds: `151001,151002`.
- Validation seeds: disjoint ranges beginning at `181001` and `181201`.
- Final evaluation seeds: `171001,171002`; these are never visible to the
  controller.
- Action catalogue: `q={1,2,4,8}`.
- Sliding window `W=8`, progress scale `0.005`, confidence coefficient `0.25`,
  `delta=0.05`, forced re-probe interval `40`, no stationary-risk Markov bias
  correction (`zeta_H=0`) because the optimized object is the registered
  finite-horizon validation risk.
- Every decision holds `q` fixed for `K` MAPPO updates, then observes one
  before/after common-random-number validation pair.  Both trajectories and
  packets are charged.
- Fixed-q baselines consume the maximum number of training updates permitted
  by the same two budgets and pay no sensing cost they do not use.
- Reporting evaluations are common measurement operations and never update a
  model or controller.

All task-specific constants, budgets, and model settings are in
`TSP-V2/experiments/marl_online_development.json`.  The task sizes deliberately
make the environment budget active and the message queue discriminate among
participation levels.

## Frozen analysis

For each task and coupling, the strong fixed baseline is the fixed-q method
with the largest mean development AUC across the two paired seeds.  Return
curves are interpolated on the registered budget-fraction axis and held at the
last observation after a method exhausts its usable integer updates.

Confirmation is authorized only if all gates pass:

1. exactly 40 unique, finite, zero-exit runs with exact source and artifact
   hashes;
2. no physical-budget violation and no final-evaluation leakage;
3. byte-identical analyzer replay;
4. at least 1% mean AUC improvement over the strong fixed baseline on **each**
   task after averaging its two coupling regimes;
5. final-return noninferiority within 2% on each task;
6. at least two participation levels used in every controller cell.

Any failure permanently stops this development design.  Runs are not retried
for scientific failure, and thresholds, seeds, tasks, budgets, constants, or
analysis are not changed after outcomes are observed.

## Compute and storage

Runs use one A30 each, scratch only, in a new root
`/scratch/jzhuangag/MARL-SDDE-TSP-V2-ONLINE-DEV-001`.  Nothing is written to
`/project`; no cleanup is authorized.  Arrays are submitted sequentially in
offsets of eight to avoid overlapping waves.
