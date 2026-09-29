# TSP-V2 online Lyapunov MARL development preregistration 002

## Scientific question

Under identical communication and actor-transition budgets, does the online
Lyapunov learning-progress controller improve return-versus-budget area under
the curve relative to the strongest fixed participation level on both
PettingZoo MPE and MaMuJoCo, while preserving final-return noninferiority?

## Frozen relationship to prior work

The stopped `DEV-001` experiment produced no scientific observation and is not
reused.
`QUAL-002` passed every real-runner path and accounting gate at commit
`ec500b0c80b45c93513ba2ae7f85de4ccf162ad8`.

This experiment changes only the qualified runtime bridge, experiment identity,
scratch root, and seed streams.
It preserves the two tasks, independent/shared coupling regimes, controller,
five comparison methods, optimization settings, physical budgets, analysis,
and mandatory scientific gates from the original development design.

## Frozen grid

The grid contains 40 unique cells:

- two tasks: MPE `simple_spread_v2` and MaMuJoCo `HalfCheetah-v2/2x3`;
- two cross-worker couplings: independent and shared;
- five methods: controller and fixed `q` in `{1,2,4,8}`;
- two fresh paired training seeds: `153001` and `153002`.

Validation seed bases are `183001` and `183201`; evaluation seeds are `173001`
and `173002`.
Training, validation, and evaluation streams are mutually disjoint and are not
reused from `DEV-001`, `QUAL-001`, or `QUAL-002`.

## Frozen mandatory gates

All 40 runs must be unique, finite, exit successfully, preserve exact dual
budget accounting, and keep evaluation unavailable to the controller.
The frozen analyzer must replay byte-identically.

For each task separately, the controller must obtain at least 1% normalized
return-AUC gain over the strongest fixed-q comparator, remain within 2% of that
comparator in final return, and select at least two distinct participation
levels in every controller cell.

Any mandatory failure stops confirmation.
No seed, task, method, controller constant, budget, analysis rule, or threshold
may be changed after observing development outcomes.

## Runtime and storage

Runs are staged sequentially in five eight-cell waves under
`/scratch/jzhuangag/MARL-SDDE-TSP-V2-ONLINE-DEV-002`.
Nothing is written to `/project`.
Each wave is verified before the next wave is submitted, and no failed payload
is retried under this identifier.
