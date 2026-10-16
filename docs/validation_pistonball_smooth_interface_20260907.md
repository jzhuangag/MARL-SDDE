## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: experiment validation
- Origin Date: 2026-09-07
- Verification Status: VALIDATED DEVELOPMENT RESULT; NOT PILOT OR FORMAL EVIDENCE
- Version Label: pistonball_smooth_interface_qualification_v1

# Smooth Pistonball learner and signed-score qualification

## Decision

Both frozen qualification gates pass. The SiLU distinct-actor
centralized-critic learner and its one-VJP signed freshness interface may be
used to design a matched controller-headroom matrix. No efficacy pilot is yet
authorized: these two development runs test necessary interfaces, not whether
the proposed scheduler beats strong equal-resource baselines.

## Provenance

- Source commit: `461a7461adeaf5c4f31b69647d9f1c1d4066260a`.
- Slurm array: `1825697_0` and `1825697_1`.
- Raw root:
  `/scratch/jzhuangag/causal-policy-freshness-icml2027/artifacts/smooth-interface-461a7461adeaf5c4f31b69647d9f1c1d4066260a`.
- Summary SHA-256:
  `4abaecf3deb12c1404629ede84360bdba89c8a15062ff65254251d4c7e283d37`.
- Fresh-run SHA-256:
  `a2729bc0c98f41c1f012ba1a9d1338e76511e821bac60861abd3d6c6ebdeedaa`.
- Scale-run SHA-256:
  `0ffce212c26ebcf7da6428e543deb846fd5c59bd56572196a128f73ed2c8da44`.

Both jobs used deterministic CUDA and completed `0:0` on A30/gpu02. Stderr
is empty. Runtime was 10:22 for the learner and 5:37 for the scale run; Slurm
peak RSS was 3,112,524 and 3,151,004 KiB respectively. Nothing was written to
`/project`.

## Learner gate

The fully fresh SiLU run used seed `79005`, 4,096 launches, 327,680 actor
transitions, zero delay, actor step `0.01`, and critic step `0.0003`.

| Actor transitions | Decentralized return |
|---:|---:|
| 0 | 1.930260 |
| 81,920 | 4.386442 |
| 163,840 | 4.780340 |
| 245,760 | 4.041419 |
| 327,680 | 4.037074 |

The terminal learning delta is `2.106814`, above the frozen threshold `2.0`.
Mean packet-gradient norm is `0.049271` and mean actor parameter drift is
`0.084344`. The gate passes, although its 0.1068 margin is not wide and does
not establish robustness across seeds.

This diagnostic spent 77,501 refresh units and 6,934,479,476 optional policy
bytes. It is an unconstrained learner upper bound, not a communication result.

## Signed-score gate

The matched ReLU/SiLU scale seed `79004` used 2,048 launches, maximum delay
eight, finite total budget 0.5 per launch, and an active dual queue.

- scored launches: 968;
- signed-favorable fraction: `0.427686`, above the frozen 0.01 threshold;
- raw signed delta range: `-1.11803e-8` to `4.63522e-8`;
- queue active fraction: `0.742769`;
- raw cache energy median: `1.19381e-5`;
- selected fraction under legacy scale: `0.698347`;
- all component reconstruction checks: pass.

The outcome-blind frozen calibration gives `V=1e8` after clipping and
`beta=83,650.1231`. The fact that `V` hits its prespecified cap shows the mixed
curvature remains numerically small. The next headroom matrix must therefore
include signed-only and cache-only/mismatch ablations; otherwise a gain could
not be attributed to the signed term.

## Scope

The result closes the ReLU implementation mismatch without erasing it. The
ReLU job `1825676` remains a failed scale gate and its raw files remain
unchanged. Seeds `79004` and `79005` are development-only and excluded from
all pilot/formal populations. No confidence interval, significance test,
generalization claim, or scheduler-performance claim is made here.
