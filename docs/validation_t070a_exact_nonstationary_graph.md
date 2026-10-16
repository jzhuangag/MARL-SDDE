# T-070A exact nonstationary collaboration-graph validation

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-27
- Verification Status: VERIFIED
- Version Label: t070a_validation_v1

## Decision

T-070A passes the exact feasibility gate. The nonstationary safe dynamic oracle
reduces aggregate geometric cumulative personalized risk by 13.9085% relative
to the cellwise best of all 2,401 time-invariant recipient-specific graphs. It
strictly improves 201 of 288 nonstationary cells (69.7917%).

This authorizes a separately preregistered observable sampled CPU pilot. It does
not authorize formal seeds, a nonlinear benchmark, GPU, or HPC4.

## Frozen provenance

- Preregistration commit: `b2533eb`
- Configuration SHA-256:
  `08a0621afc6c28fca79c64eaa1c9e5cb89e3bf2c00a4b6d4533b2b01827ebb43`
- Cells SHA-256:
  `CF00672BBE4F58155C32E858B9E9ABA3FB5C42EC711A9A58B9B414D5668C0B7F`
- Summary SHA-256:
  `B3C25D4D319BAF33FD5FFD5FBA3CE6CD387C611117C0BFF05DDFF042E4CFD14F`
- Scientific rows: 432
- Static graph risks evaluated: 1,037,232
- First execution wall time: 12.54 seconds on local CPU

No sampled trajectory, formal seed, Slurm job, GPU job, or HPC4 artifact was
created.

## Primary result

| Comparison | Geometric ratio | Improvement | Strict cells |
|---|---:|---:|---:|
| All nonstationary / best static graph | 0.860915 | 13.9085% | 69.7917% |
| Single switch / best static graph | 0.903497 | 9.6503% | descriptive |
| Alternating / best static graph | 0.820340 | 17.9660% | descriptive |
| Stationary control / best static graph | 1.073361 | -7.3361% | descriptive |

The stationary control is deliberately unfavorable to the dynamic oracle
because the oracle pays for probes while the best static graph receives no
sensing charge. Its 7.3361% loss is evidence that the positive nonstationary
result is not caused by a universally privileged dynamic comparator.

## Delay and phase structure

| Delay in blocks | Dynamic improvement over best static graph |
|---:|---:|
| 0 | 16.1404% |
| 1 | 13.4521% |
| 3 | 12.0833% |

All registered delay groups remain positive. Benefit decreases with delay, as
expected for a tracking mechanism using stale donors.

The target-scale groups reveal a material phase boundary:

| Target scale | Improvement | Strict-cell fraction |
|---:|---:|---:|
| 0.1 | -4.2070% | 40.6250% |
| 0.3 | 11.2099% | 68.7500% |
| 0.6 | 31.0365% | 100.0000% |

Thus T-070A supports a conditional claim: dynamic graph tracking is valuable
when collaboration affinities move far enough to overcome probing and delayed
adaptation costs. It does not support uniform dominance under arbitrarily small
drift.

Every nonstationary cell changes its selected graph at least once. All 432
registered schedule-change boundaries induce a graph change. Changes also
occur at 742 of 1,008 non-change decision boundaries, reflecting ordinary
bias--variance evolution in addition to exogenous target shifts.

## Gate ledger

| Gate | Result | Evidence |
|---|---:|---|
| R1 finite exact workload | pass | 432 cells; 1,037,232 static risks |
| R2 charged-shadow checkpoint safety | pass | coverage 1.0 |
| R3 environment charging | pass | exactly 240 transitions per cell |
| R4 message budget | pass | at most 18 units per cell |
| R5 aggregate nonstationary gain >= 5% | pass | 13.9085% |
| R6 strict nonstationary cells >= 60% | pass | 69.7917% |
| R7 each changing schedule gain >= 3% | pass | 9.6503%, 17.9660% |
| R8 nonnegative gain for every delay | pass | 12.0833%--16.1404% |
| R9 graph changes in >= 60% cells | pass | 100% |
| R10 stationary improvement <= 5% | pass | -7.3361% |
| R11 at least eight static graphs selected | pass | 22 |
| R12 byte-identical clean rerun | pass | both artifact hashes identical |

## Correctness and reproducibility

The targeted suite has nine tests. In addition to input, workload, charging,
retargeting, and safety checks, it compares the batched exact implementation
against scalar block propagation for both a stationary local graph and a
nonlocal graph crossing an abrupt target change.

The complete repository experiment suite passed under the project `.venv`
(the established `ust2` runtime): 599 passed and 7 skipped in 118.43 seconds.

A clean execution in a separate reproduction directory produced byte-identical
`cells.csv` and `summary.json`. The reproduction directory remains ignored and
is not scientific evidence beyond the deterministic hash check.

## Interpretation boundary

T-070A is an exact outcome-aware oracle ceiling in a scalar affine Markov-TD
class. It establishes that the redesigned problem contains substantial dynamic
collaboration value against a strong static graph. It does not yet establish
that a low-complexity observable controller can identify the graph soon enough.
That is the mandatory question for the next sampled CPU pilot.
