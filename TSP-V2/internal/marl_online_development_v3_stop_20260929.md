# TSP-V2-MARL-ONLINE-DEV-003 stop decision

## Decision

`DEV-003` **fails the frozen multi-environment development gates**.
Confirmation is not authorized, no confirmation seeds are registered, and no
result from this development run may enter the TSP manuscript as positive
evidence.

This is a scientific performance failure, not an execution or accounting
failure.  All 40 registered runs completed and passed the payload, leakage,
dual-budget, ledger, source, and checksum audits.  The unchanged controller
nevertheless failed to outperform the cellwise strongest fixed participation
baseline on both registered tasks.

## Provenance and execution

- preregistration commit: `a470a723362047f042b7fcdc586bc718edf4c4c4`;
- scientific implementation commit:
  `a6f6716c388da74255c21f46463a8bba6f596b88`;
- scratch root: `/scratch/jzhuangag/MARL-SDDE-TSP-V2-ONLINE-DEV-003`;
- Slurm arrays: `1919049`, `1919185`, `1919511`, `1920096`, `1920675`;
- all 40 array elements: `COMPLETED`, exit `0:0`;
- `/project` writes: none;
- retries, changed seeds, changed thresholds, and dropped environments: none.

All artifact-local `SHA256SUMS` manifests passed.  The staged Git tree remained
clean at the preregistration commit, every run reported pinned HARL commit
`b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`, and independent replay of every
ledger reconstructed the exact message and environment charges.

## Frozen analysis

The analyzer was executed twice into disjoint directories.  The CSV, JSON, and
stdout were byte-identical.  Replay-1 hashes are:

- `summary.csv`: `781ca8974d428785b66cb0193f4b90615a768d8ce104be9fd9ab574289bf5519`;
- `decision.json`: `608b1616c35e4fd974b4798602c18e38fd6136de3e5de2979e7e655075a30afb`;
- `stdout.txt`: `608b1616c35e4fd974b4798602c18e38fd6136de3e5de2979e7e655075a30afb`.

Task-level results relative to each cell's strongest fixed `q` were:

| Task | Mean normalized-AUC change | Mean final-return change | Frozen requirement |
|---|---:|---:|---:|
| MPE `simple_spread_v2` | -8.9092% | -16.1018% | AUC >= +1%, final >= -2% |
| MaMuJoCo `HalfCheetah-v2/2x3` | -26.8388% | -37.7684% | AUC >= +1%, final >= -2% |

Cellwise AUC changes were `-18.5087%` (MPE/independent), `+0.6903%`
(MPE/shared), `-52.1616%` (MaMuJoCo/independent), and `-1.5159%`
(MaMuJoCo/shared).  The controller used multiple participation levels in every
cell, so failure is not explained by a degenerate single-action policy.

## Gate ledger

| Gate | Result |
|---|---:|
| Complete 40-cell population | pass |
| Exact dual-budget accounting | pass |
| No evaluation leakage | pass |
| Byte-identical analyzer replay | pass |
| Nondegenerate online selection in every cell | pass |
| Each task mean AUC gain at least 1% | **fail** |
| Each task mean final-return change at least -2% | **fail** |

## Consequence

The current learning-progress controller is stopped for this two-environment
claim.  It must not be rescued by rerunning these seeds, weakening the gates,
dropping an environment, or comparing only against a weaker fixed baseline.
Any future redesign requires a new scientific mechanism, a new identifier, an
outcome-free implementation commit, and fresh development seeds.
