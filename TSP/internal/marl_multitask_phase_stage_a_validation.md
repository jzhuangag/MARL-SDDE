# TSP multi-task participation phase: Stage-A validation

## Material Passport

- Artifact type: experiment validation record
- Experiment: `TSP-MARL-MULTITASK-PHASE-A-001`
- Preregistered source commit: `43f3d3ced19a127167da62c42dd6d6af580b84dd`
- Analysis status: `VERIFIED`
- Scientific role: development-only endpoint headroom scan
- Decision: `STOP`

## Execution and provenance

The frozen 32-run lattice completed without a scientific or infrastructure failure.
It contains two tasks, two budget rays, two dependence regimes, two endpoint participation levels, and two development seeds.
The four Slurm arrays were `1890024`, `1890470`, `1891145`, and `1891496`; every array element completed with exit code `0:0`.
All artifacts remain under `/scratch/jzhuangag/MARL-SDDE-TSP-MULTITASK-PHASE-A-001`; the experiment wrote nothing to `/project`.

Every result directory passed its recorded `SHA256SUMS` check.
All 32 metadata records were unique, finite, associated with the frozen experiment identifier and HARL commit `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`, and satisfied both registered budgets.
The analyzer was executed twice from the frozen source and produced byte-identical gate and cell files.
The final replay-enabled hashes are:

- `cells.csv`: `a5f3f299a33e46ab7842a1eb66b266d4015fe518fd6e09ca5f78213093ef1dd3`
- `gates.json`: `446c94909b6669e3b422668f8d492ea8a2d05ef00aa623483c1960dff5c6a405`

## Frozen gate results

| Gate | Result |
|---|---:|
| Finite, complete, exact accounting, clean upstream | Pass |
| At least two oracle actions per task | Pass |
| Message-binding independent/shared reversal per task | Pass |
| Byte-identical analysis replay | Pass |
| Oracle headroom at least 2% for every task | **Fail** |

The task-level results are:

| Task | Strong static endpoint | Oracle actions | Oracle headroom |
|---|---:|---:|---:|
| MPE Speaker--Listener | `q=8` | `{1,8}` | `0.311959%` |
| MaMuJoCo HalfCheetah `2x3` | `q=8` | `{1,8}` | `3.793433%` |

Both tasks exhibit the registered message-binding regime reversal: independent sampling selects `q=8`, whereas shared sampling selects `q=1`.
MaMuJoCo also clears the task-level 2% headroom threshold.
Speaker--Listener does not: its cellwise endpoint oracle improves over the task-level strong static endpoint by only `0.311959%` on average.

## Decision

The preregistered conjunction requires the headroom gate to hold for every task.
Consequently, Stage A stops and does not authorize the planned `q=2,4` completion or the dual-budget Lyapunov controller on this frozen two-task suite.
The tasks, seeds, budget rays, thresholds, and endpoint results must not be changed retrospectively to reverse this decision.

The MaMuJoCo result is preserved as positive development evidence of dependence-sensitive participation value, but it is not a controller result and is not confirmatory evidence.
No Stage-B, confirmation, or additional GPU run was started.

