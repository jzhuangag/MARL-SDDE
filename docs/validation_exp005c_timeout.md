# Validation report: EXP-005C timeout

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-29
- Verification Status: UNVERIFIED
- Version Label: timeout_report_v1

## Validation Report

- **Source**: EXP-005C-sparse-dynamic-controller
- **Overall Confidence**: RED_FLAG

The 64-seed primary experiment did not complete and generated no primary
artifacts. The registered scientific gates cannot be evaluated.

### Execution finding

| Item | Result |
|---|---|
| Registered timeout | 15 minutes |
| Observed wrapper timeout | approximately 2,228 seconds |
| Child process | identified and terminated |
| Primary result files | none |
| Automatic retry | none |
| Verdict | CANNOT_VERIFY |

The 4-seed smoke result cannot substitute for the registered 64-seed analysis.
Its failed gates are useful only for computational diagnosis.

### Fallacy scan

- **Coverage**: 11/11 types checked

| Fallacy | Severity | Finding |
|---|---|---|
| Simpson's paradox | NOTE | No primary subgroup results exist. |
| Ecological fallacy | NOTE | No population inference is made. |
| Berkson's paradox | NOTE | No completed primary comparison exists. |
| Collider bias | NOTE | No covariate adjustment is used. |
| Base-rate neglect | NOTE | Not applicable. |
| Regression to the mean | NOTE | Not applicable. |
| Survivorship bias | RED_FLAG | Treating only the completed smoke run as evidence would be a form of completion-based selection; it is explicitly prohibited. |
| Look-elsewhere effect | NOTE | No primary result was searched or selected. |
| Garden of forking paths | NOTE | Registered settings remain unchanged after timeout. |
| Correlation versus causation | NOTE | No empirical causal claim is made. |
| Reverse causality | NOTE | Not applicable. |

### Reproducibility

- **Method**: not run;
- **Verdict**: CANNOT_VERIFY.

### Validated conclusion

EXP-005C currently provides no scientific evidence for or against the sparse
dynamic controller. It establishes only that the registered implementation
requires performance diagnosis or a separately authorized longer execution.
