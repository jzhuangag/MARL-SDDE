## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: experiment validation and stop decision
- Origin Date: 2026-09-07
- Verification Status: VALIDATED NEGATIVE DEVELOPMENT RESULT
- Version Label: pistonball_controller_headroom_v1

# Pistonball one-VJP controller headroom validation

## Decision

The frozen controller-headroom matrix fails and does not authorize an efficacy
pilot. The one-cross-policy-VJP signed Lyapunov configuration is stopped.
Twenty-two of twenty-two runs completed, every validity and resource invariant
passed, and the proposed controller took nontrivial actions at acceptable
runtime. Nevertheless it lost to the strongest equal-budget baseline in both
development seeds, failed AUC no-harm, and did not show separate value from its
signed-only and cache-only components.

This decision does not change thresholds, add seeds, or rerun outcomes. It
also does not claim that policy staleness is irrelevant. Complete freshness
has opposite effects across the two seeds, showing that blindly maximizing
freshness is itself unsafe. The failed part is the present linearized score and
cache-debt combination as a reliable solution to that signed decision.

## Provenance

- Frozen source commit: `52253f88f8b6b4964db5a9dabbfcbc08424b1b01`.
- Development-only seeds: `79006`, `79007`.
- Slurm batches: `1825716_0`--`1825716_7`,
  `1825732_8`--`1825732_15`, and `1825748_16`--`1825748_21`.
- Raw root:
  `/scratch/jzhuangag/causal-policy-freshness-icml2027/artifacts/controller-headroom-52253f88f8b6b4964db5a9dabbfcbc08424b1b01`.
- Summary SHA-256:
  `8650d642b0c4b31a210f70c9601cc74fe19330da7dfb9f3bc7e2f9ffe3da41fa`.

The initial 22-task and 11-task submissions were rejected atomically by
`QOSMaxSubmitJobPerUserLimit`; neither created a Slurm job. The unchanged
matrix was then split into arrays of eight, eight and six tasks. All tasks
completed `0:0`, all stderr files are empty, and peak RSS remained below
3.24 GiB. No output was written to `/project`.

## Frozen gate result

| Gate | Result | Value |
|---|---:|---|
| H1 validity/accounting | pass | all 22 finite, matched, drained, within cap |
| H2 complete/no-refresh headroom | **fail** | paired `-12.5887`, `+1.1341`; mean `-5.7273` |
| H3 nontrivial signed spend | pass | 1,551 and 1,268 of 2,048 units |
| H4 gain over best strong baseline | **fail** | paired `-4.8819`, `-0.9369` |
| H5 normalized headroom | **fail** | undefined favorable ratio because mean ceiling is negative |
| H6 component value | **fail** | signed-cache `+0.9321`; signed-signed-only `-1.3851` |
| H7 AUC no-harm | **fail** | `-2.0251` versus best strong AUC |
| H8 runtime | pass | 1.1132 times no-refresh |

The strongest mean terminal baseline is no-refresh. The strongest mean AUC
baseline is the terminal-cap complete burst. The proposed mean terminal
return is `-3.8401`, compared with no-refresh `-0.9306`, signed-only `-2.4550`,
cache-only `-4.7721`, and full refresh `-6.6579`.

## Paired outcomes

| Seed | Signed | Signed-only | Cache-only | No refresh | Full refresh |
|---:|---:|---:|---:|---:|---:|
| 79006 | 4.8619 | 7.7306 | 2.2581 | 9.7438 | -2.8449 |
| 79007 | -12.5420 | -12.6406 | -11.8023 | -11.6051 | -10.4710 |

The large paired heterogeneity is scientifically important. In seed 79006,
complete freshness is much worse than preserving stale caches; in seed 79007
it is modestly better. A controller that merely reduces cache mismatch cannot
be uniformly useful. The proposed signed term improves on cache-only on
average, but the composite cache energy erases the advantage of signed-only,
and neither variant beats no-refresh.

## Scope and next admissible question

No pilot, formal seeds, confidence interval, or ICML efficacy claim is
authorized. The stopped one-VJP finite-displacement approximation must not be
rescued by changing `V`, `beta`, budget, seeds, or gates against these results.

The discrete Lyapunov theorem accepts any predictable simultaneous estimate
of each candidate action drift; it does not require the failed Taylor/VJP
approximation. One outcome-free repair remains scientifically motivated:
compute the owner-gradient alignment under each actual one-edge cached policy
counterfactual inside the sparse causal cone. This removes the unreliable
finite-displacement linearization at `O(Delta)` reverse evaluations rather
than one. It may proceed only after an implementation/overhead audit and a
new frozen headroom design. If that exact sparse score still lacks headroom,
the Pistonball policy-cache mainline should stop rather than be retuned.
