# TSP-V2 learning-progress sensor G0-B validation

## Material Passport

- Experiment: `TSP-V2-MARL-PROGRESS-G0B`
- Preregistration commit: `b863673eee82f03d62a47f6e7b0c7e0bf8687e80`
- Slurm array: `1915670`
- Remote root: `/scratch/jzhuangag/MARL-SDDE-TSP-V2-PROGRESS-G0B`
- Decision: **authorize coefficient-certificate design**
- Manuscript performance evidence: **none**

## Execution and integrity

All eight registered array elements completed with exit code `0:0`.
The source checkout remained clean at the preregistration commit.
Every cell produced one registered sensor JSON and one complete
`SHA256SUMS` manifest; all eight manifests verified without error.
The nonempty PettingZoo stderr files contain only the pinned wrapper's
observation-space and action-space deprecation warnings.
A fatal-pattern scan found no traceback, exception, nonfinite-value, memory,
or storage failure in a completed payload.

Across all 32 candidate branches:

- initial actor and critic hashes were identical within each cell;
- every reward trace contained exactly 24 finite observations;
- PettingZoo charges equaled `24 * (100 + 25 q)` messages and `24 * 25`
  environment ticks;
- MaMuJoCo charges equaled `24 * (800 + 200 q)` messages and `24 * 200`
  environment ticks;
- no evaluation return was computed.

## Frozen gate result

The diagnostic selections were:

| Environment | Coupling | Seed 140201 | Seed 140202 |
|---|---|---:|---:|
| PettingZoo MPE | independent | 1 | 4 |
| PettingZoo MPE | shared | 2 | 4 |
| MaMuJoCo | independent | 2 | 8 |
| MaMuJoCo | shared | 4 | 4 |

All frozen mandatory gates passed.
In particular, the MaMuJoCo cells exhibited both broad and interior
participation signals, while all PettingZoo traces were finite and
nonconstant.

The frozen analyzer was executed twice into separate files.
The files were byte-identical with SHA-256
`0efe46cb236174684fedc2ef3402cdc075c51639880c4f9f4849d9f16ecdb7de`.
The replay-enabled decision is `authorize-certificate-design`.

## Scope of the decision

G0-B establishes that fully charged short training branches expose a
task-dependent participation signal on two standard multi-agent learning
environments and that the proposed interface is executable and reproducible.
It does not establish that raw training reward is a valid Lyapunov
coefficient certificate, that the diagnostic selection improves final return,
or that the V2 controller is ready for manuscript inclusion.

The next authorized stage must replace the diagnostic late-reward selector by
a separately preregistered coefficient-certificate layer using disjoint
validation observations and fresh seeds.

