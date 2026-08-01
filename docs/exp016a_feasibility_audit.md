# EXP-016A feasibility audit for Amendment 1

This is a preregistration feasibility audit. It contains no trajectory,
pilot, formal, HPC4, GPU, result-row, or scientific outcome generation.

## Frozen inputs

- Original preregistration commit:
  `592986466ff55281914dd76a4faad7338ea91914`
- Original configuration SHA-256:
  `bb3ab51bc64d4ee334e7c5da6b6e7a4e7ffd303692abb6a5e48d06e48bb9baf5`
- Positive base scenarios: 54
- Above-`B_S` budget points per scenario: 2
- Cells per identification direction: 108
- Pilot seeds per cell: 64
- Familywise alpha: 0.05
- Directional delta: 0.025

The original preregistration files remain immutable provenance. Amendment 1
supersedes only the analysis rules and feasibility decision.

## G6 design-level red flag

The original G6 required a per-cell directional Clopper-Pearson/Holm upper
bound <= `0.025`. With 108 cells per direction, the smallest Holm tail level is
`alpha/108 = 0.00046296296296296298`. Even if a cell observes zero errors, the
sample count needed is

```text
ceil(log(alpha/108) / log(1 - 0.025)) = 304
```

Without multiplicity correction the corresponding zero-error requirement is
119 seeds. Therefore the frozen 64
pilot seeds cannot pass the original per-cell G6 gate, even under perfect
zero-error observations. This is a design-level `RED_FLAG`, not something to
confirm by running a doomed pilot.

## Revised feasibility findings

- G4 practical-effect subset: 108 of
  108 high above-`B_S` cells.
- G8 learning-value-active subset: 0.
- G9 delay-active high cells: 36.
- G10 message-active high cells: 0.
- G10 environment-active high cells: 0.
- Minimum G4 expected relative gain over all high cells:
  `0.14865002914813402`.
- Minimum G5 analytic margin to epsilon:
  `-0.015104078773273008`.

Because the learning-value-active subset is empty, revised G8 is infeasible
under the current frozen policy definitions. Amendment 1 selects stop gate
decision **B**: stop the ICML adaptation-cost main line and do not run the
EXP-016A pilot.

## Workload decision

The seed count remains 64. No fresh pilot seeds are added because adding
seeds cannot repair an empty novelty subset or the original G6 role mismatch.

- Expanded cells: 550
- Policies: 10
- Estimated rows/trajectories: 352000
- Estimated single-process CPU hours: `1.1733333333333333`
- Estimated peak memory GB: `1.5`
- Estimated disk GB: `0.49280000000000002`

Local CPU would have remained within the preregistered resource envelope, but
the pilot is not authorized after this audit.
