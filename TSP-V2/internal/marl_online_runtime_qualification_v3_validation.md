# TSP-V2 audited HARL runtime qualification validation

## Decision

`TSP-V2-MARL-ONLINE-QUAL-003` **passes every mandatory runtime and accounting
gate**.  It authorizes a separately preregistered two-environment development
experiment.  This qualification contains no performance evidence and none of
its returns may be used for tuning, inference, figures, or manuscript claims.

## Provenance

- execution commit: `f773bae520403dd973de14adc7d7b80f6d934675`;
- scientific implementation commit: `a6f6716c388da74255c21f46463a8bba6f596b88`;
- Slurm array: `1918632_[0-7]`;
- scratch root: `/scratch/jzhuangag/MARL-SDDE-TSP-V2-ONLINE-QUAL-003`;
- pinned HARL commit in every payload:
  `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`;
- `/project` writes: none.

All eight array elements completed with exit code `0:0`.  Elements 0--7 ran on
`gpu10`, `gpu01`, `gpu01`, `gpu10`, `gpu07`, `gpu01`, `gpu01`, and `gpu07`,
respectively.  No fatal signature was found in the sixteen stdout/stderr logs,
the staged Git tree remained clean, and every per-run `SHA256SUMS` manifest
verified.

## Gate results

| Gate | Result |
|---|---:|
| Eight unique task/coupling/method/seed payloads | pass |
| Finite progress and metadata | pass |
| Evaluation return hidden from controller | pass |
| Controller ledger length and indices exactly 9 and 1--9 | pass |
| Statistical window remains 8 while cumulative counts sum to 9 | pass |
| Ledger-selected counts equal metadata counts | pass |
| Per-decision message and environment formulas replay exactly | pass |
| Remaining-budget recurrence and terminal totals replay exactly | pass |
| Fixed comparator dual-budget update count replays exactly | pass |
| Clean source commit, fatal-log scan, and SHA-256 manifests | pass |

The controller exercised nondegenerate selections in all four controller
payloads.  The cumulative count vectors were:

- MPE/independent: `(q1,q2,q4,q8)=(2,3,2,2)`;
- MPE/shared: `(6,1,1,1)`;
- MaMuJoCo/independent: `(1,2,2,4)`;
- MaMuJoCo/shared: `(2,2,3,2)`.

These counts are runtime evidence only.  In particular, they do not establish
that any choice improves return.

## Authorization

A new scientific identifier may now be preregistered with fresh seeds.  It
must compare the unchanged online controller against all fixed
`q in {1,2,4,8}` under the same two physical budgets on both MPE
`simple_spread_v2` and MaMuJoCo `HalfCheetah-v2/2x3`, with independent and
shared coupling.  Both tasks must pass their frozen task-level performance
gates before confirmation is authorized; environments may not be dropped
after observing outcomes.
