# T-083A formal confirmation audit

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-31
- Verification Status: ANALYZED
- Version Label: validation_v1

## Decision

The frozen all-gates decision is **FAIL**. Preserve both completed runs; do not rerun to obtain a passing execution time, change gates, or overwrite their summaries. F1–F11 passed in both runs. F12 passed originally and failed on reproduction. F13 failed. F14 now passes in this separate audit record; the frozen summaries retain their original pending/false F14 field.

This is an execution/reproducibility-gate failure, not a reversal of the measured scientific effect. Neither fact removes the other. No nonlinear/GPU escalation is authorized by this report.

## Provenance and reproducibility

Preregistration HEAD: `61431bbc95f48bdeb81206a0111db91f04cd1fb5`.
Original and reproduction directories are respectively `experiments/dependence_delay_linear/results/t083a_formal_end_block_confirmation` and the same path suffixed `_reproduction`.

| Check | Original | Reproduction | Verdict |
|---|---:|---:|---|
| Runtime, seconds | 1034.4742100000003 | 2787.0133623 | Reproduction exceeds frozen 2100-second limit |
| endpoints.csv SHA-256 | C10975BE82199FE82EF2853B18BD97A04AF0978A5E6CDAF6BC772CE4FC61114B | same | MATCH |
| cells.csv SHA-256 | 61761F0B9472778B5073D2473A8FC9A1C994E269698E61F309AD6FDC63AF1EB6 | same | MATCH |
| summary.json SHA-256 | DD932FA9DBB5E203B3DE027E74A813FED737FE7CE9386C09DB4593E52A7CC228 | 6DD44415AF441F96A580F3A54BFAE4DEAA03FB8D9181A5A067C9D45922417BCD | MISMATCH |

The complete metrics dictionaries match. Only `gates` (F12) and `all_pre_reproduction_gates_pass` differ. Timing is recorded separately but affects the summary through F12. The frozen F13 nevertheless requires all three artifacts to match, so it fails. The cause of the runtime difference has not been established.

Full regression command (repository root on PYTHONPATH): `.venv/Scripts/python.exe -m pytest experiments -q`. Result: **663 passed, 7 skipped in 109.31s**. No scientific trajectories were generated for this audit.

## Scientific interpretation

Overall confidence: **CAUTION**. There are 128 formal seeds, 55,296 endpoints, 96 primary cells and 336 controls. Primary controller/static error ratio is 0.7604943789 (23.950562% improvement); 90/96 cells strictly improve. These are frozen descriptive aggregate comparisons, not newly computed p-values or confidence intervals.

Stationary/local ratio is 1.0513604475 (5.136% cost); low-signal ratio is 1.0064857884; high-temporal ratio is 0.9849091396. Universal no-harm is not supported. The primary population has temporal correlation zero; its strong gain does not establish an equally strong general Markov result. The static reference is the registered optimized best-found reference, not a proven globally optimal static graph.

### Statistical fallacy scan

Coverage: 11/11 checked at report/aggregate level, not a new raw-data inferential analysis.

| Type | Finding |
|---|---|
| Simpson | Controls differ from primary; do not pool away stationary cost. |
| Ecological | Cell aggregates do not guarantee every agent or trajectory improves. |
| Berkson | Frozen selected population limits external generalization. |
| Collider | No new covariate-conditioned analysis introduced here. |
| Base rate | Diagnostic prevalence claims are not applicable. |
| Regression to mean | Independent seeds retained; no extreme-seed selection permitted. |
| Survivorship | Both completed runs and failed gates retained. |
| Look-elsewhere | All frozen populations retained; earlier exploratory successes are not formal evidence. |
| Forking paths | No post-result gate or seed changes; historical calibration remains exploratory. |
| Correlation/causation | Controlled simulator comparison does not establish real-world RL effectiveness. |
| Reverse causality | No observational directional claim introduced. |

## Next stage: theory–implementation audit, not another efficacy run

Verify possible completed-block double counting in NoiseCertificate; establish or reject confidence coverage; derive the relation between empirical debt and true personalized risk; audit Markov block-generation law and complexity. Preserve frozen controller code. Corrections require explicit versioned provenance and cannot retroactively validate this formal run. A new CPU experiment needs a justified theory interface, feasibility checks and its own preregistration, not a new identifier to evade failed gates.
