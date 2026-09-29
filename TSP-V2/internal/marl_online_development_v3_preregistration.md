# EXP TSP-V2-MARL-ONLINE-DEV-003 preregistration

## Scientific question

Does the unchanged online Lyapunov learning-progress controller improve the
equal-resource return curve over the strongest fixed participation level in
**both** a discrete cooperative MARL task and a continuous-control CTDE task?

This is a fresh-seed development experiment.  It is not confirmation and
cannot enter the manuscript as final evidence.  Qualification returns from
`QUAL-003` are excluded from every scientific statistic and were not used to
change the controller, tasks, budgets, methods, or gates.

## Frozen implementation

- controller implementation: commit
  `a6f6716c388da74255c21f46463a8bba6f596b88`;
- preregistration base: commit
  `6f1c6ab05004d4aab7f7cc195d629354fa03315d`;
- runner SHA-256:
  `a27a78dc6a71244c59862fe3f1edddf7588437ef4833ba3d85e56801f7aa1347`;
- audited controller SHA-256:
  `0452e7bfdc4e06e1c3c3814b08f58301333f851b972336ac19fef72b7e04562d`;
- analyzer SHA-256:
  `9754bd054632409eaae9da3cb9f95fd589fbd5c2a0ad6622cee5bc0f3c8f259b`.

No algorithmic constant changed after `DEV-002`; the only scientific-code
change is a passive cumulative ledger that is proved by unit tests not to
alter the action sequence or budget state.

## Frozen population

The 40 cells are the Cartesian product of:

- tasks: PettingZoo MPE `simple_spread_v2` and MaMuJoCo
  `HalfCheetah-v2/2x3`;
- cross-agent couplings: independent and shared;
- methods: proposed controller and fixed `q in {1,2,4,8}`;
- fresh common-random-number training seeds: `253001`, `253002`.

Validation streams start at `283001` and evaluation streams use `273001` and
`273002`; they are disjoint from training and from all stopped or qualified
experiments.  Every method receives the same two physical message and actor
transition budgets.  Fixed comparators do not pay for observations they do not
consume; the controller pays for both disjoint validation rollouts at every
decision.

The task hyperparameters, 100 controller decisions, candidate catalogue,
window, optimism coefficient, queue weight, learning rates, and budgets are
frozen in `TSP-V2/slurm/marl_online_development_v3_a30.sbatch` and are identical
to the prospective `DEV-002` scientific design.

## Frozen analysis and gates

For every task/coupling cell, the comparator is the fixed `q` with the largest
mean normalized return AUC across the two development seeds.  The proposed
method must therefore beat a cellwise strong fixed baseline, not only `q=1` or
`q=8`.

All mandatory gates must pass:

1. all 40 payloads complete, remain finite, have unique registered mappings,
   verify their hashes, and pass exact ledger-based dual-budget replay;
2. evaluation returns are never used by the controller;
3. each task's mean relative normalized-AUC improvement over its strong fixed
   baseline is at least `1%` after averaging its independent/shared cells;
4. each task's mean final-return relative change is at least `-2%`;
5. the controller selects at least two participation levels in every
   task/coupling cell;
6. two clean analyzer executions produce byte-identical CSV and JSON outputs.

Failure of any mandatory gate stops confirmation without retry, environment
removal, seed replacement, threshold change, or reinterpretation.  Only a
complete pass on **both environments** authorizes a separate preregistration
with new confirmation seeds and a paired one-sided test.  Only passing
confirmation results may be added to the TSP manuscript.

## Execution discipline

Artifacts are scratch-only under
`/scratch/jzhuangag/MARL-SDDE-TSP-V2-ONLINE-DEV-003`.  The five array waves use
offsets `0,8,16,24,32`, must never overlap, and may not be retried after a
scientific or payload failure.  `/project` writes and cleanup are forbidden.
