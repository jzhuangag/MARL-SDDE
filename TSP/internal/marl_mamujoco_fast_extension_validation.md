# MaMuJoCo fast-extension validation

## Material Passport

- Experiment: `TSP-MARL-MAMUJOCO-DEV-001`
- Role: prospective development; not confirmation evidence
- Frozen preregistration commit: `85029286ad217b06a889739730cbce6fccd69774`
- Executed source commit: `91fad2e1f6de9e85d8ab9db676a651bab4fee50b`
- G0 validation commit: `bb2ead61fc994ff7e22f8e304f655aff071bc9b7`
- Pinned HARL commit: `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`
- Task: MaMuJoCo `HalfCheetah-v2/2x3`, MAPPO, shared parameters
- Decision: **stop; do not preregister confirmation and do not admit this extension to the manuscript**

## Execution and integrity

The outcome-free G0 job `1892331` passed before the development run.  The
five sequential A30 arrays were `1892338` (offset 0), `1892992` (offset 8),
`1893599` (offset 16), `1894555` (offset 24), and `1895223` (offset 32).
All 40 registered runs completed with exit code `0:0`.  Every top-level run
directory passed its recorded `SHA256SUMS`; bounded log scans found no Python
traceback, CUDA out-of-memory event, missing file, exhausted storage,
assertion failure, segmentation fault, or killed process.

The audit recovered exactly 32 unique fixed-policy records and eight unique
controller records.  Each record used the registered task, coupling, seed,
budget, and clean pinned upstream commit.  Fixed methods obeyed the exact
message/environment horizon induced by their participation level.  Controller
runs charged 38,400 probe messages and 3,200 probe environment ticks before
training, and every complete run remained within both physical budgets.

Two independent executions of the frozen analyzer produced byte-identical
cell CSV and gate JSON files.  Their SHA-256 values were respectively
`96a33111c13762074414af3a6e21104774d9104e2c0a23319ca50f532088e19f`
and `5d49316f3b940440c1aa7f530f96ff20fc35c36e7de569c11605e483770d0215`.
The replay-enabled final gate JSON differs only by setting the registered
replay gate to true and has SHA-256
`9307ce411292deb31caa239792d68889783f1ce69de1f7caf7dc2d2e882d4972`.

## Scientific result

The controller made a genuine regime-dependent decision: its four
independent runs selected `q = [8,4,8,8]`, whereas all four shared runs
selected `q=1`.  Averaged over both regimes, its return AUC exceeded the
strongest single fixed policy (`q=8`) by `3.5504%`, clearing the registered
`1%` mixture-value gate.  Probe-message use was `0.768%` of the message budget
and maximum scalar-selection overhead was `3.31e-7`, so both cost gates also
passed.

The mandatory regime-wise envelope gate failed:

| coupling | best fixed method | best fixed return AUC | controller return AUC | controller gap |
|---|---:|---:|---:|---:|
| independent | `q=8` | 3865.3990 | 3638.3539 | -5.8738% |
| shared | `q=2` | 2018.4318 | 1750.6442 | -13.2671% |

The registered tolerance was `-2%` in **each** regime.  Consequently the
overall development decision is `stop`, even though the adaptive controller
beats every single fixed `q` in the equally weighted mixture.  The task,
controller, seeds, thresholds, and analysis were not changed after observing
the result.  No confirmation seeds were registered or run.

Machine-readable outputs are
`marl_mamujoco_fast_extension_cells.csv` and
`marl_mamujoco_fast_extension_gates.json`.  Raw artifacts remain under
`/scratch/jzhuangag/MARL-SDDE-TSP-MAMUJOCO-DEV-001`; no experiment output was
written to `/project`.

## Consequence for the TSP manuscript

This development extension is excluded from the manuscript.  It does not
alter the existing positive controlled Markov-TD and preregistered PettingZoo
evidence.  The result may inform a future, separately identified controller
design, but it cannot be used to tune and reconfirm this frozen experiment.
