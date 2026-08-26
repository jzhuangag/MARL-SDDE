# T-069 recipient-specific static baseline validation

## Decision

T-069 is a reproducible mandatory-gate failure.  Q1--Q3 and Q5--Q10 pass after
the external clean-rerun audit, but Q4 fails: the fully charged dynamic safe
oracle improves geometric terminal risk over the cellwise best
recipient-specific static vector by 2.0093%, below the frozen 3% threshold.
No sampled safe-mixing pilot, formal experiment, standard benchmark, GPU job,
or HPC4 job is authorized.

The failure is informative rather than degenerate.  Dynamic safe mixing is
strictly better in 58.9506% of cells, has nonnegative aggregate improvement in
all four heterogeneity groups and all three delay groups, and selects against a
static class containing 1,296 vectors.  The recipient-specific static oracle
itself improves 3.3810% over the common-scalar oracle and selects 25 distinct
vectors.  Thus both recipient personalization and temporal control carry value,
but the latter is not large enough under this stationary model to justify the
next experiment stage.

## Frozen provenance

- T-068A result commit: `07db085`.
- Independent T-069 preregistration commit: `923787f`.
- Configuration SHA-256:
  `A33AB8E7D3FED00D4033AB909AE844A22CE03F6522E02CDFA3D12F601F978981`.
- Runner SHA-256:
  `22048C53560A0E2C18D101949B6102204C58117BF8BE99449190BC8EC90418F4`.
- Batched exact core SHA-256:
  `E425E0EB972FC01AB43C9DB6F8229D18161BEA9219492B6FB64994C80F260250`.
- Frozen T-068A cells SHA-256:
  `F79F9D34BEA89EEDD243E3DB555E51133CD29227DE2A9CC05D551D0A0DA99BD8`.
- Workload: 648 cells and 839,808 evaluated recipient-vector risks.
- Hardware: local CPU only.

## Gate ledger

| Gate | Frozen requirement | Observed | Status |
|---|---|---:|---|
| Q1 | 648 cells and 839,808 finite risks | exact | pass |
| Q2 | common-alpha subset reproduces T-068A | exact | pass |
| Q3 | recipient-static no worse than common scalar cellwise | 100% | pass |
| Q4 | dynamic aggregate improvement at least 3% | 2.0093% | **fail** |
| Q5 | dynamic strict improvement in at least 40% cells | 58.9506% | pass |
| Q6 | nonnegative in at least three heterogeneity groups | four of four | pass |
| Q7 | nonnegative in every delay group | three of three | pass |
| Q8 | at least eight selected static vectors | 25 | pass |
| Q9 | frozen source hash and cell identity | exact | pass |
| Q10 | clean rerun byte-identical | both artifacts exact | pass |

## Group structure

Dynamic improvement over recipient-static is 2.9357%, 0.8659%, 1.6488%, and
2.5736% for heterogeneity 0, 0.05, 0.2, and 0.5.  By delay it is 3.0694%,
0.0153%, and 2.9133% for delays 0, 1, and 3.  The near-zero value at delay one
prevents a robust monotone delay story and must not be hidden by aggregation.

## Scientific consequence

The stationary safe-mixing idea remains proof-feasible, but its present
experimental value is too small for the intended ICML mainline.  The T-068A
5.3224% gain cannot be reported as pure online adaptation because 3.3810% is
explained by recipient-specific static personalization.

The next allowed work is theory/design only.  A defensible continuation must
make time variation intrinsic to the research question, for example a bounded-
variation sequence of heterogeneous Markov environments, and prove dynamic
regret and charged safety prospectively.  It may not retrospectively alter the
stationary T-068A/T-069 cells or gates.

## Reproduction

The clean reproduction is byte-identical:

| Artifact | SHA-256 |
|---|---|
| `cells.csv` | `8419990B7398BE30C9B3B6271F8FE54FC68CEEDF7E881FAAD1156056705DE683` |
| `summary.json` | `D8CBB18CBA1E02ECA604C615F7E078D942ED1BE797777FB9995D743C5EE64ED1` |

The T-069 targeted audit contains five passing tests, including exact
agreement between the batched common-alpha subset and the original T-068A
moment runner.  The final full repository regression completed with
`590 passed, 7 skipped in 691.72 s`.
