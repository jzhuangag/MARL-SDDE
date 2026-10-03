# EXP-016B CPU pilot validation

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-01
- Verification Status: VERIFIED
- Version Label: exp016b_validation_v1

## Validation Report

- Source: EXP-016B frozen configuration `6dfdf87521700c2ddae9b81947e0ecc01ee33ebcf5fcda34b09e9e3c3f7f7ee5`
- Preregistration commit: `7dc4da8618186d9b4514c5ad956e4c4e09b85e00`
- Execution: local CPU only; no GPU, HPC4, `/project`, formal seed, or MARL benchmark
- Rows: 1,376,256 from 96 frozen pilot seeds
- Overall confidence: SOLID for the registered pilot mechanism
- Decision: P1--P12 passed. This permits a separate formal preregistration; it does not itself create formal evidence.

### Primary statistical findings

The primary estimand is paired `risk(information_only) - risk(learning_aware)` over the registered Layer-A practical-Z population. The simultaneous one-sided calibration uses the frozen Bonferroni critical value `3.0233414397391534` at familywise alpha `0.01`.

| Population | Paired mean | Simultaneous lower bound | Relative to always-all | Result |
|---|---:|---:|---:|---|
| Layer A primary | 0.1767004 | 0.1541207 | 55.5126% | directional and practical gates pass |
| Layer B affine TD | 0.0677304 | 0.0510166 | 10.1985% | same direction as Layer A |
| Delay-active | 0.1669302 | 0.1435861 | 51.0670% | pass |
| Message-binding | 0.1982870 | 0.1618949 | 72.8906% | pass |
| Environment-binding | 0.1383203 | 0.0981163 | 40.9693% | pass |

Scenario-level practical-Z coverage is `77/96 = 0.8020833`, above the frozen `0.60` gate. All 8,640 neutral-Z seed rows were retained and excluded from rescue of the primary estimand. The Layer-B catalogue retained all 56 registered tasks.

### Gate audit

| Gate | Result | Evidence |
|---|---|---|
| P1 | PASS | 1,376,256 expected rows; all finite; zero dual-budget violations; complete eight-policy grid |
| P2 | PASS | no below-`B_id` run claimed a reliable certificate |
| P3 | PASS | information-only probes and learning-aware falls back in every registered Z cell |
| P4 | PASS | Layer-A point estimate, simultaneous lower bound, and 3% practical threshold all pass |
| P5 | PASS | scenario coverage 80.2083% |
| P6 | PASS | all neutral-Z cells retained |
| P7 | PASS | at/above `B_value`, convergent policies have identical probe/action plans under the same CRN path |
| P8 | PASS | empirical crossover occurs at the registered bracket boundary; no forced post-convergence gap |
| P9 | PASS | 96/96 finite scenarios satisfy theorem-facing `S_mean <= epsilon_safe`; minimum slack `6.5392e-06` |
| P10 | PASS | delay-active, message-binding, and environment-binding subsets are nonempty and directional |
| P11 | PASS | Layer B has the same positive direction and retains every registered task |
| P12 | PASS | four core artifacts are byte-identical under a clean same-seed rerun |

### Reproducibility

- Method: deterministic same-code, same-environment, same-seed rerun
- Verdict: REPRODUCIBLE
- Raw metrics SHA-256: `1596bd6995d316e1cb48388d1859c323dc74effc5d478b1edf50e9a205c8d052`
- Cell summary SHA-256: `e43b37a2e3be3c12fe91dd8edb137b96d285ee014ce7aa73aa2e360b63f9950f`
- Scenario summary SHA-256: `8ac4bb0dd6b5bc82aa1e5f7d3d5764f77cf037b535ed13ca3c5d7f64705359d4`
- Core results SHA-256: `8717762d017cbf0b58a955d641ffc648c5dd116f2d7333034de6a5a6c91a01aa`
- Final runner SHA-256: `fcdcb36e6b79622b9e2925b5dca8d36ad771f2a0447f5b813a3d3b52b90af5b0`
- Final analyzer SHA-256: `dfa7e5557fe28cf2d5c4f4b18b025d59dd60bfcdb808855e08600a0ef215b2be`

The full comparison is machine-readable in `docs/exp016b_reproduction_audit.json`. Raw pilot and reproduction CSV files remain local and are excluded from Git.

A final taint audit split the information-only rule into an independent
three-input interface `(scale, B_id, probe)`, with no `B_value`, safety,
downstream-risk, regime, or outcome input. A complete post-audit 96-seed run
again produced 1,376,256 rows and the identical raw metrics SHA-256
`1596bd6995d316e1cb48388d1859c323dc74effc5d478b1edf50e9a205c8d052`.

### Fallacy scan

Coverage: 11/11 checked.

| Fallacy | Finding |
|---|---|
| Simpson's paradox | No reversal: aggregate, delay, message, environment, and Layer-B registered directions are positive. |
| Ecological fallacy | Claims remain at the registered scenario/task-family level; no individual-agent inference is made. |
| Berkson's paradox | Static scenario selection preceded outcomes and used only T-018 metadata and SHA ordering. |
| Collider bias | No outcome-dependent conditioning or covariate adjustment is used. |
| Base-rate neglect | Identification errors are reported within the balanced registered low/high regime design; no deployment prevalence claim is made. |
| Regression to the mean | Paired CRN seed blocks are not selected by extreme observed performance. |
| Survivorship bias | All registered rows, policies, tasks, seeds, neutral cells, and censored descriptive cells are retained. |
| Look-elsewhere effect | P1--P12, primary population, direction, practical threshold, and simultaneous correction were frozen before pilot outcomes. |
| Garden of forking paths | The scientific analysis path is frozen. The executable implementation was completed and smoke-tested after preregistration, so this pilot is not promoted to formal evidence; the implementation must be committed before fresh formal seeds are assigned. |
| Correlation versus causation | The result is a randomized simulation contrast under a specified mechanism, not a claim about uncontrolled real-world systems. |
| Reverse causality | Not applicable to the intervention-defined policy comparison. |

### Integrity boundary

The pilot runner was implemented after the static preregistration, as explicitly allowed by its handoff. Non-scientific smoke seeds were used to verify finiteness, budget charging, delay-queue stability, and mechanism direction; they are excluded from all pilot aggregates. Because no independent implementation-freeze commit preceded the pilot, the conservative next step is a new formal preregistration commit that fixes the runner hash, analysis hash, environment, and fresh formal seeds before any formal run.
