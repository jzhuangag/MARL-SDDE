# EXP-017A nonlinear GPU pilot validation

## Decision

The preregistered implementation-only A30 pilot completed successfully, but
the scientific progression decision is **negative**. Mandatory gates G7, G9,
G11, and G12 failed. Under the frozen all-gates rule, no formal seed list is
registered and no formal EXP-017A run is authorized. The pilot seeds remain
excluded from any later confirmatory population.

This result changes none of the completed EXP-016B code, seeds, gates,
artifacts, or claims. It also does not support unrestricted unknown-mixing
adaptation, occupation optimality, or general nonlinear MARL.

## Frozen scope

- Preregistration commit: `17a4c324763ccf38e2738240dc1c67608cad8337`
- Configuration SHA-256:
  `28c1c24181c6de02fd0b48e7f420c3ea46887024ccb0551ccef9597f109ae5ea`
- Pilot seeds: `20550101`, `20550102`
- Tasks: Gymnasium `CartPole-v1` and `Acrobot-v1`, fixed stochastic
  behavior policies, neural semi-gradient TD prediction
- Population: 1,584 endpoints and 77,847 trajectory rows, covering both
  tasks, two registered mixing certificates, three correlations, three delay
  traces, two budgets, and all eleven policies
- Dependence: marginal-preserving common/private complete-trajectory coupling;
  no artificial observation-noise advantage
- Evidence status: implementation-only GPU pilot

The exact runner, analyzer, configuration, registry, tasks, seeds, budgets,
correlations, delays, baselines, metrics, and gates were committed and pushed
before submission. The remote worktree remained clean at that commit.

## HPC4 execution

Storage was checked before submission. `/home` had 133 GB available,
`/scratch` 234 GB, and `/project` 3.4 TB at the initial preflight; inode use
was 1%. Active code, dependency overlay, logs, and outputs were placed under
`/scratch/jzhuangag`. No experiment output was written to `/project`; only the
already-existing frozen Python environment there was read.

The first `sbatch` attempt was rejected before job creation because the Slurm
account was not specified. This was submission plumbing only and generated no
scientific output. The unchanged frozen payload was then submitted with
`--account=vincentlau`:

```text
sbatch --parsable --account=vincentlau --export=ALL,EXP017A_CODE_ROOT=/scratch/jzhuangag/MARL-SDDE-exp017a-17a4c32,EXP017A_RUN_ROOT=/scratch/jzhuangag/exp017a-pilot-17a4c32,EXP017A_PYTHON=/project/vincentlau/jzhuangag/MARL-SDDE/envs/exp014a-py39/bin/python,EXP017A_EXPECTED_COMMIT=17a4c324763ccf38e2738240dc1c67608cad8337,PYTHONPATH=/scratch/jzhuangag/exp017a-overlay-17a4c32 /scratch/jzhuangag/MARL-SDDE-exp017a-17a4c32/slurm/exp017a_pilot_a30.sbatch
```

Slurm array `1685696` produced task job IDs `1685696` and `1685697`. Both ran
on `gpu10` with one NVIDIA A30, 16 CPUs, and 64 GB requested memory. Both
started at `2026-08-01T18:32:04`, completed at `19:10:33`, ran for `00:38:29`,
and exited `0:0`. Peak RSS was 1,565,644 KiB and 1,595,076 KiB. Both stderr
files are empty; both seed-level `sha256sum -c SHA256SUMS` checks passed.

## Frozen gate results

| Gate | Result | Recorded diagnostic |
|---|---:|---|
| G1 finite and dual-budget valid | pass | all registered endpoints finite and within both budgets |
| G2 exact population | pass | all 1,584 endpoints present without duplicate keys |
| G3 mixing certificates | pass | both registered known-mixing certificates valid |
| G4 marginal construction | pass | bank metadata and common/private construction recorded |
| G5 information-only taint audit | pass | no forbidden outcome or oracle input found |
| G6 communication matching | pass | budgets equal within every paired comparison block |
| G7 correlation response | **fail** | median selected `q` was 1 at both rho=0 and rho=.9; strict decrease absent |
| G8 delay response | pass | median `b` was 4 for both zero and nonzero delay; nondecrease holds |
| G9 primary improvement | **fail** | learning/information geometric ratio 0.9999275; improvement 0.00725%, below frozen 2% threshold |
| G10 task consistency | pass | Acrobot ratio 0.9998551; CartPole ratio 1.0; both within registered directional tolerance |
| G11 fixed-q noninferiority/tail | **fail** | geometric ratio 5.0006 and CVaR90 ratio 25.0450 versus pilot-selected best fixed-q |
| G12 overhead/metric completeness | **fail** | metrics complete, but controller wall fraction 0.50137 exceeds frozen 0.10 ceiling |

The failure pattern is substantive rather than a numerical crash. The
learning-aware controller collapsed to `q=1`, did not respond to the
correlation range, produced essentially no primary gain over the
information-only baseline, and was far worse than the pilot-selected fixed-q
envelope. Its Python-level finite-table selection also consumed about half of
measured wall time. These findings identify implementation/design weaknesses
for a future separately named study; they do not justify post-hoc repair of
EXP-017A.

## Artifacts and reproducibility

The active artifact root is
`/scratch/jzhuangag/exp017a-pilot-17a4c32` (about 46 MiB including runtime
cache directories). The clean source clone is
`/scratch/jzhuangag/MARL-SDDE-exp017a-17a4c32`; logs are under
`/scratch/jzhuangag/exp017a_logs`. The compact committed summary is
`exp017a_pilot_summary.json`; complete hashes and job metadata are in
`exp017a_pilot_reproduction_audit.json`.

Core combined-artifact SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| `endpoints.csv` | `bc241c772d20b76c5f42f72bd8a5523bda2ba225e113811e695dd840007191f0` |
| `trajectories.csv` | `5f0979038facf4e22536720f683cb37e350f166e2203202c30dd4d8f5007bce4` |
| `cell_summary.csv` | `2fec45646bcd4e571a632076b3d2a98a3b4cac9139bdadd75484237623029264` |
| `pilot_selected_best_fixed_q.csv` | `6302ead5fdcaef6e2aa418522f6c27d180ad297c7bcdb2c45ec4fd8faa3b48ee` |
| `pilot_summary.json` | `ebeb9ce78fdf2b0486641cd572f06de7481031d0eb7067c1a669dfb3ebc0fd9f` |

An independent second invocation of the frozen analyzer reproduced all five
core hashes exactly.

The HPC4 targeted suite passed `10/10`. A whole-repository collection attempt
under the reused Python 3.9 environment stopped on pre-existing files using
Python 3.10 `X | None` annotations; it did not expose an EXP-017A failure. The
local whole-repository run had 216 passes and three existing EXP-016B
raw-byte registry failures caused by Windows CRLF working-tree conversion;
the frozen Git blob hashes were verified and EXP-016B was not modified.

## Claim boundary and next decision

EXP-017A supplies only a reproducible negative implementation pilot on two
standard fixed-policy Markov prediction tasks under known or independently
certified separated mixing. It authorizes no positive nonlinear benchmark
claim and no formal experiment. Any future controller repair must receive a
new experiment identifier and a new outcome-free preregistration; the current
gates, seeds, analysis population, and negative record remain frozen.
