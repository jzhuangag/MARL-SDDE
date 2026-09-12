# Analytic strict-shield activation scan

## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: analytic feasibility validation
- Origin Date: 2026-08-31
- Verification Status: ANALYZED
- Version Label: audit_v1

## Frozen scope and result

Grid, diagnostic and tests were committed before scan execution in `ca33223`. This is a transparent toy public-model grid, not a claim to reproduce the registered formal scenario distribution. No formal endpoints or seeds were accessed. It contains 108 scenarios and 1,728 block-level analytic probabilities. Target scales are in units of marginal noise standard deviation (variance fixed at one). Mixing is supplied exactly in both model and interval calculation, not estimated from data. Failure probability .05 is allocated across four agents and 16 blocks within each scenario, not across all scenarios as a single deployment.

| Target scale | Scenarios | Mean of scenario mean probabilities | Maximum scenario mean probability |
|---|---:|---:|---:|
| 0 | 36 | 0.0010404220164992322 | 0.013231273229031396 |
| 0.5 | 36 | 0.0010509718459436837 | 0.013231301415040704 |
| 2 | 36 | 0.013915531393251909 | 0.15382643632068416 |

These probabilities measure |shadow-center|>radius. They characterize whether any unrestricted scalar candidate could have strictly negative robust excess; graph restrictions can only reduce that opportunity. They are not measured controller activation frequencies, performance gains, probability of ever activating, or a proof that every confidence-based method fails. The maximum above is a scenario time-average, not the maximum individual block probability. Constant and switching labels coincide at target scale zero; the frozen duplicate controls remain retained.

The diagnostic is unfavorable for directly deploying this strict interval shield in the examined settings. Even exact mixing knowledge does not make its activation broad. No success gate was registered; no post-hoc numerical gate is introduced. The decision is to withhold efficacy experiments pending a sharper risk-certificate/design analysis, not to declare a formal statistical failure or tune the grid until favorable.

## Reproducibility and integrity

Command: `.venv/Scripts/python.exe -m experiments.dependence_delay_linear.run_shield_activation_audit` from repository root. Two CPU executions returned exit 0 and exactly equal parsed outputs. Complete block-level outputs are retained in shield_activation_scan_results.json; no sampled scientific trajectory or ignored experiment result directory was generated. Five diagnostic unit tests passed in 2.81s before preregistration commit. Deterministic formula evaluation does not estimate confidence intervals or p-values; no significance interpretation is made.

Plan SHA-256: `339F1AD5FDECBC125E152C66B8B2BF945BB5C2C1ABF8C8819FF38475DAEB67EF`.
Results SHA-256: `977BC96F71D2B587EFFD1CBE286D75AAC11FC908CBAD313847485F3E994CC419`.

Statistical fallacy checklist: 11/11 reviewed at scope level. Simpson/ecological: aggregates are not per-block guarantees. Selection/Berkson: toy grid scope stated. Collider: no conditioning on achieved outcomes. Base-rate: marginal event defined. Regression-to-mean: no selected sampled extremes. Survivorship: every registered row retained. Look-elsewhere/forking paths: pre-scan grid retained, descriptive only. Causation/reverse causality: no empirical or real-world causal inference. No scientific outcome, gate, or frozen controller was altered.

## Next bounded task

Derive a directly certified paired-risk statistic that exploits cancellation of shared observation noise, rather than requiring a full target interval each block. Audit its predictability/Markov bias and failure-event contribution first. Compare its information requirements analytically with the current interval shield. If that route also cannot provide a valid nontrivial guarantee, record the obstruction and reassess the safety contract; do not simply increase risk allowances or restart old experiments.
