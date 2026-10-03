## Material Passport

- Experiment: T-078 delay-extremes persistent-certificate calibration
- Verification status: VERIFIED negative calibration
- Data scope: 288 theory-selected cells, 32 reused design seeds, 9,216 endpoints
- Claim scope: architecture evidence only; not an independent pilot
- Frozen commit: `234b65f8017bd402c4020373d10e718c5cb4129f`

# Validation decision

T-078 is a mandatory-gate failure. R3, R7, R8, and R13 fail; no new-seed
pilot, formal run, nonlinear benchmark, GPU, or HPC4 execution is authorized.
Thresholds, cells, seeds, and controller constants remain unchanged.

The persistent continuous-QP controller retains aggregate value: it improves
nonstationary cumulative risk by 8.1237%, both schedule families by 4.7436% and
11.3838%, and delay 0/3 by 8.7300%/7.5133%. Resource, certificate-direction,
stationary-control, complexity, runtime, and full-test gates pass.

Broad safety does not pass. Strict cell coverage is 52.6042% against 55%; the
temporal-correlation 0.9 ratio is 1.01084 against 1.01; and target-scale 0.1 is
1.06428 against 1.06. These are small threshold misses but cannot be rounded or
reclassified. The result supports a phase-dependent benefit, not uniform
dominance over the strong static graph.

## Reproducibility

The original eight-worker run completed in 471.38 seconds and the clean rerun
in 150.83 seconds. Endpoint and cell CSV files are byte-identical. The saved
summary is not byte-identical because `runtime_seconds` was included in the
JSON; after removing that timing field, every scientific metric and gate is
identical. The frozen R13 nevertheless remains failed. Full regression passes:
`635 passed, 7 skipped in 202.42 s`.

## Statistical-fallacy scan

Coverage is 11/11. The delay-extreme selection was frozen from the stability
question and did not use outcomes. All cells, seeds, gates, and negative strata
are retained, avoiding Berkson, survivorship, look-elsewhere, and forking-path
selection. Aggregates are decomposed by schedule, delay, signal scale, and
temporal correlation, so the direction heterogeneity is explicit rather than
hidden by Simpson aggregation. There is no agent-to-population inference,
collider adjustment, diagnostic base-rate claim, extreme-score preselection,
or observational causal claim. Target schedules precede actions.

## Scientific next step

Further controller tuning on these seeds is stopped. The defensible research
program is now a phase theorem: characterize the observable region in which
dynamic continuous collaboration beats every static graph, the low-SNR or
near-nonmixing region in which adaptation cannot be uniformly no-harm, and the
opportunity-cost/safety lower bound between them. The positive algorithmic
claim must be restricted to the identifiable separated class and paired with
the lower bound outside it.
