# TSP-MARL-XTASK-DEV-001 validation report

## Scope and stopping decision

TSP-MARL-XTASK-DEV-001 is the outcome-free development transfer of the frozen
Lyapunov probe-then-commit controller from PettingZoo MPE `simple_spread_v2` to
the communication-conditioned `simple_reference_v2` task.  The controller,
candidate actions, budgets, model, optimizer settings, seeds, evaluation rule,
and mandatory gates were fixed at commit
`804b1f834964c3afb4dd197ec178eb4910534c2c` before any development trajectory
was generated.

The experiment passes seven of nine mandatory gates.  It fails the frozen
shared-stream practical-gain floor and the equal-mixture practical-gain floor.
The decision is therefore **STOP**: no confirmation seeds are registered, no
confirmation run is authorized, and this development result is not manuscript
evidence.  The two failed thresholds are retained without rounding or
modification.

## Frozen-gate results

| Gate | Frozen threshold | Observed | Status |
|---|---:|---:|---:|
| Complete, finite, exact accounting, clean upstream | required | 24/24 | pass |
| Independent selection of `q=8` | at least 3/4 | 4/4 | pass |
| Shared selection of `q=1` | at least 3/4 | 4/4 | pass |
| Independent controller AUC vs. fixed `q=8` | at least -2% | -0.0922% | pass |
| Shared controller AUC vs. fixed `q=8` | at least +3% | +2.6944% | **fail** |
| Equal-weight mixture AUC vs. fixed `q=8` | at least +1.5% | +1.3011% | **fail** |
| Probe message fraction | at most 1% | 0.768% | pass |
| Controller wall-clock overhead fraction | at most 5% | `9.7784e-7` | pass |
| Marginal-preserving coupling audit | required | pass | pass |

The controller's observable dependence decision transfers perfectly in these
four development seeds, and the independent-stream cost remains small.  The
return advantage on the new task is nevertheless below both preregistered
practical-effect requirements.  Mechanism selection alone is not sufficient
to promote a task into the paper.

## Execution and integrity

The 24 runs were split into three immutable eight-cell arrays on NVIDIA A30
GPUs:

| Slurm array | Cells | Method block | Result |
|---|---:|---|---|
| `1849778` | 0--7 | controller | 8/8 `COMPLETED 0:0` |
| `1849960` | 8--15 | fixed `q=1` | 8/8 `COMPLETED 0:0` |
| `1850262` | 16--23 | fixed `q=8` | 8/8 `COMPLETED 0:0` |

All 24 per-cell `SHA256SUMS` files pass.  Their ordered digest-of-digests is
`df802b8dd4896af8b7b8028ed3d44a85a0898504129c4ee35715bb56e91462fc`.
The fatal-log scan contains zero matches.  The frozen analyzer was executed in
two fresh output directories; gate JSON, curve CSV, and PDF are byte-identical
between analysis and replay.

- Gate JSON SHA-256:
  `c468eaf2ee5012f7ea81124b70666a3ae99685d25cb89e7e9437e613c146e16e`.
- Curve CSV SHA-256:
  `7212f2d04ad767723d64286c6c5b62269c46e21658cbf62a6c461b9925108c93`.
- Replayed PDF SHA-256:
  `8c5506bf7204df6471287f80dcf3a68ad9355f0eb4564f2328ea345528709e01`.

Artifacts remain under
`/scratch/jzhuangag/MARL-SDDE-TSP-MARL-XTASK-DEV-001` (408 MB at validation).
No output was written to `/project` or `/home`.  The preceding two-cell smoke
job `1849775` remains an integration check only and is excluded from every
scientific statistic.

## Manuscript consequence

The confirmed `simple_spread_v2` result and its compact Fig. 5 remain
unchanged.  No `simple_reference_v2` number, curve, or claim is admitted to the
paper.  The compact figure layout is retained because it improves page use
without changing the confirmed evidence.
