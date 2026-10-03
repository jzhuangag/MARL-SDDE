# EXP-016A preregistration Amendment 1

Amendment 1 supersedes the EXP-016A analysis rules while preserving the
original preregistration files as immutable provenance.

## Binding hashes

- Original configuration SHA-256:
  `bb3ab51bc64d4ee334e7c5da6b6e7a4e7ffd303692abb6a5e48d06e48bb9baf5`
- Amendment configuration SHA-256:
  `a6312a4769457c3d73aedea60f9c3523a2860d6fc0499ad1ccdb6124188412d0`

The amendment hash binds the original configuration hash, the Amendment 1 file
set, the revised gate table payload, and the static feasibility audit payload.

## Reason for amendment

The original G6 gate was a mandatory per-cell rare-event confidence gate. With
108 cells per direction and 64 pilot seeds per cell, even zero observed errors
would give a Holm-floor upper bound above `0.025`. The frozen calculation is
recorded in `exp016a_feasibility_audit.json` and marks this as a design-level
`RED_FLAG`.

The amendment separates deterministic theorem/runtime compliance from
empirical calibration:

- G6a is a mandatory deterministic audit of thresholds, likelihoods, stopping
  boundaries, `q/b`, delay accounting, budget accounting, and hidden-state
  leakage.
- G6b is mandatory aggregate directional empirical calibration, with per-cell
  intervals reported descriptively and a single-cell anomaly threshold of
  observed error rate `>0.10`.

## Outcome-free active subsets

Amendment 1 also prevents identical policy paths from being forced to produce
nonzero novelty or mechanism effects. G8, G9, and G10 operate only on
outcome-free active subsets where the frozen intended plans differ before any
trajectory is generated.

The current frozen definitions produce a G8 learning-value-active subset of
size `0`.
Because this subset is empty, the EXP-016A pilot is not authorized. This
selects stop gate decision **B** from the preregistered feasibility audit.

## Authorization

No scientific trajectory, pilot, formal run, HPC4 job, GPU job, result row, or
scientific outcome was generated. The next stage should redesign the novelty
comparison or stop the adaptation-cost ICML main line; it should not run the
current EXP-016A pilot.
