# TSP-V2 audited HARL runtime qualification plan

## Purpose

`TSP-V2-MARL-ONLINE-QUAL-003` is an outcome-free software qualification for
the audit-preserving online Lyapunov runner.  It is not development,
confirmation, tuning, or manuscript evidence.  In particular, evaluation
returns from this stage must never be interpreted as method performance.

The scientific implementation is commit
`a6f6716c388da74255c21f46463a8bba6f596b88`.  The new observer leaves the
frozen sliding-window decision state unchanged while maintaining a separate
cumulative selection counter and per-decision physical-resource ledger.
Frozen SHA-256 values are:

- runner: `a27a78dc6a71244c59862fe3f1edddf7588437ef4833ba3d85e56801f7aa1347`;
- audited controller: `0452e7bfdc4e06e1c3c3814b08f58301333f851b972336ac19fef72b7e04562d`;
- exact-accounting analyzer: `9754bd054632409eaae9da3cb9f95fd589fbd5c2a0ad6622cee5bc0f3c8f259b`.

## Frozen grid

The eight jobs cross:

- PettingZoo MPE `simple_spread_v2` and MaMuJoCo `HalfCheetah-v2/2x3`;
- independent and shared cross-agent coupling;
- the online controller and `fixed_q1` execution branches.

Controller jobs execute nine decisions, deliberately exceeding the statistical
window of eight.  This is the minimal runtime test that can distinguish the
cumulative ledger from the window-truncated histories.  Seeds, task settings,
budgets, and the command line are frozen in
`TSP-V2/slurm/marl_online_runtime_qualification_v3_a30.sbatch`.

All artifacts are written only below
`/scratch/jzhuangag/MARL-SDDE-TSP-V2-ONLINE-QUAL-003`; `/project` is forbidden.

## Mandatory gates

All eight jobs must satisfy every gate:

1. exit code `0:0`, no fatal log signature, and the registered clean Git tree;
2. exact unique task/coupling/method/seed mapping;
3. finite progress and metadata with no controller use of evaluation returns;
4. controller ledger length exactly nine and decision indices exactly 1--9;
5. cumulative selected counts reconstructed from the ledger, including counts
   beyond the window length;
6. every message and environment charge reconstructed independently from the
   frozen formulas, with exact remaining-budget recurrence;
7. fixed comparator update count reconstructed from both budgets;
8. every artifact passes its local `SHA256SUMS` manifest.

Any failure permanently stops `QUAL-003` without retry or gate modification.
Only a complete pass authorizes a separately preregistered development run with
fresh scientific seeds.  No performance gate is evaluated here.
