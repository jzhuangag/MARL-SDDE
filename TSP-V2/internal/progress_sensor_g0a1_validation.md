# TSP-V2 learning-progress sensor G0-A1 validation

## Material Passport

- Scientific experiment: `TSP-V2-MARL-PROGRESS-G0`
- Operational attempt: `G0-A1`
- Frozen scientific source: `2eaa373f3283d70bf4475173ab1365a2e617e2cc`
- Operational amendment source: `d3248eccd46717f65b6d0a6810bd80e0364de414`
- Slurm array: `1915136`
- Remote root: `/scratch/jzhuangag/MARL-SDDE-TSP-V2-PROGRESS-G0A1`
- Decision: **stop**
- Certificate-layer authorization: **false**
- Manuscript evidence: **none**

## Execution result

The operational repair succeeded: Slurm created every stdout and stderr file
and invoked all eight payloads.
The four MaMuJoCo tasks completed with exit code `0:0`.
The four PettingZoo MPE tasks failed with exit code `1:0` while constructing
the environment because the pinned runtime does not contain the `supersuit`
package.
The failure occurred before an MPE candidate branch produced a registered
sensor record.

This is a payload dependency failure rather than a learning-progress result.
Nevertheless, it makes the preregistered eight-cell interface incomplete, so
the mandatory completeness and PettingZoo gates fail.
The frozen no-retry rule is retained: array `1915136` is not resubmitted, the
runtime is not changed in place, and no task, seed, candidate, threshold, or
gate is altered.

## Completed-cell verification

All four completed MaMuJoCo records passed their stored `SHA256SUMS` files.
Across the 16 completed candidate branches:

- the candidate initializations were identical within each cell;
- every reward trace contained exactly 24 finite entries;
- every message charge equaled `24 * (800 + 200 q)`;
- every environment charge equaled `24 * 200 = 4800`;
- no evaluation return was computed;
- all four successful stderr files were empty;
- a fatal-pattern scan of the successful logs was empty;
- the source checkout was clean at the operational amendment commit.

The diagnostic selections in the completed MaMuJoCo subset were `q=8` and
`q=4` under independent coupling, and `q=4` and `q=2` under shared coupling.
These incomplete-subset diagnostics are not performance evidence and do not
override the overall stop decision.

## Frozen analysis replay

The frozen analyzer found four of eight expected records and returned `stop`.
Two independent analyzer executions were byte-identical with SHA-256
`5b295dc498657e3182293b51e65d464b87b9c0a5b941ff8c6361dabf42084331`.
The replay gate itself passed, while `complete_unique_cells`, the combined
interface gate, and `pettingzoo_nonconstant_signal` failed because the MPE
records do not exist.

The G0 sensor qualification is therefore terminally incomplete under its
frozen rules.
No coefficient-certificate experiment is authorized from this attempt.

