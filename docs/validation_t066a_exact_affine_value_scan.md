# T-066A exact delayed-affine value-scan validation

## Decision

T-066A is a preregistered partial success and mandatory-gate failure.  S1--S4,
S6--S8 pass, but S5 fails: the cellwise joint oracle strictly beats both
one-dimensional restrictions in 16.3580% of cells, below the frozen 30%
threshold.  Therefore the planned stationary sampled affine-TD pilot is not
authorized and was not created or run.

No threshold, task, cell, budget, or action was changed after the independent
preregistration commit.  The result does not invalidate the discrete
Lyapunov certificate or the existence of participation value; it rejects the
stronger claim that broad stationary-cell gains generally require adapting
both controls.

## Provenance and reproducibility

- Foundation result commit: `86ff92d7b0c9b83e197efdc4c29c47ccab2c3290`
- Independent T-066A preregistration commit:
  `b31b8abf3b764ee226934963f009e9f283f01da0`
- Frozen configuration SHA-256:
  `45E871C481D9AF463BF55F515B99C883EA6E6006735195B91A5F9BA6A5003CB5`
- Workload: 648 cells, 42 actions per cell, 27,216 exact action rows.
- Execution: local CPU; no random seed, GPU, HPC4, or sampled outcome.

Clean reproduction is byte-identical:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `actions.csv` | 3,886,578 | `0DDD0BB556EAD5F7C68FD4BB9CD1AE00689E4538BB19F3BF77C5C2A8E5FEEA3B` |
| `cells.csv` | 128,868 | `C51C31507DC61018F7B07C8EBFD0A3AC1B0DA584FCAA2659E7213A52B1415551` |
| `summary.json` | 391,961 | `2E4B2E6FA92350F1629BF36FF82BC95F5B6B846CF8192419EB15374E7276A871` |

As in T-065A, the single-run summary writes the external reproduction gate as
false; the independent hash comparison above establishes S8.

## Gate ledger

| Gate | Threshold | Result | Status |
|---|---:|---:|---|
| S1 common certificate | all 9 task-delay pairs | all positive | pass |
| S2 exact validity/accounting | 27,216 finite valid rows | 100% | pass |
| S3 aggregate oracle improvement | >=10% | 23.3523% | pass |
| S4 strict cell improvement | >=60% | 80.8642% | pass |
| S5 joint beats both 1-D restrictions | >=30% | 16.3580% | **fail** |
| S6 action diversity | >=2 q and eta per budget regime | 5--6 q, 6--7 eta | pass |
| S7 corrected environment horizon | divide by q | implemented/tested | pass |
| S8 clean reproduction | exact | exact | pass |

Task-level aggregate improvement is 21.75--26.40%, and both budget regimes
are positive (18.85% message-binding; 27.60% environment-binding).  The S5
failure is therefore not caused by one broken task or budget regime.

## What the failure reveals

Joint separation is structured rather than absent.  It reaches 36.1111% in
the high-noise cells and the same 36.1111% in the low-initial-error cells, but
only 5.0926% in low noise/high initial error.  This is the expected
signal-to-noise phase: far from the solution, adapting gain alone often
captures most value; near the noise floor, participation and gain become
coupled.

This observation cannot rescue S5 and is not a new retrospective gate.  It
does identify the correct next scientific question: a real online learning
trajectory moves endogenously from contraction-dominated high signal to
noise-dominated low signal.  A controller that adapts both actions across that
within-run phase change may have value even though a broad collection of
stationary cells does not require both actions simultaneously.

## Next authorized work

Only outcome-free theory/design work for a new dynamic-phase identifier is
authorized.  Before any sampled run it must prove or exactly certify that a
single trajectory crosses both action phases, compare against strong q-only,
eta-only, and fixed-schedule baselines under identical sensor costs, and
freeze a nontrivial oracle ceiling.  T-066A itself remains a failure and its
stationary sampled successor remains stopped.
