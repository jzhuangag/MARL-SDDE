# EXP-012B validation: unknown-baseline kernel certificate

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Mode: validate
- Verification Status: VERIFIED
- Formal execution: 128 fresh seeds in each of nine scenarios
- Reproducibility verdict: REPRODUCIBLE

## Decision

**PASS.**  Both validity gates and all five scientific gates pass.  This is the
last registered CPU certificate milestone: continuous observations, hidden
sharing masks, and an unknown independent-similarity baseline are handled
simultaneously.

## Formal results

| Gate | Threshold | Result |
|---|---:|---:|
| Joint anytime coverage | at least 97.5% | 100% |
| Conditional action safety | all updating radii \(<1\) | pass; maximum 0.994286 |
| Correlation ordering | non-decreasing \(\rho^+\) | pass |
| Fast-mixing identification | fixed three-part gate | pass: 0.171, 0.626, 0.994 |
| Baseline learning | fixed lower/upper gate | pass: minimum 0.0612 |
| Persistent probe nonvacuity | at least 50 probes | pass: minimum median 85 |
| Participation response | weak everywhere, one strict | pass |

For zero mixing eigenvalue, the controller selects \(q=8,2,1\) at
\(\rho=0,.5,.9\) when \(D=0\), and \(q=2,1,1\) when \(D=2\).  Thus
correlation remains actionable under a nonzero delay in the continuous-state
test.

The true periodic-kernel baseline is 0.14194.  Its fast-mixing lower confidence
bound is approximately 0.061--0.063.  At mixing eigenvalues .8 and .96 the
lower bound remains zero, while the correlation upper certificate stays safe.
This is the expected confidence/mixing penalty and prevents a uniform
sharpness claim.

## Reproducibility

All four core artifacts reproduce byte-for-byte:

| Artifact | SHA-256 |
|---|---|
| `kernel_latent_summary.png` | `B4FACE744878AEF71688224C367446204F1DFCE9E2F12DA3327916FEF6525882` |
| `metrics.csv` | `45184CBDB87D1D324EA90EF9B1C8B3096D0A50B34845529F150244D8F47EDD6D` |
| `summary.json` | `6E83E9C42C3901A2E2B79BE949DB8BF71FBE49B7401EE7258313B0DF8FB8BEC9` |
| `traces.csv` | `046E2FF2165EE1903E6A7CF67198D9245C02B3F8E2018397659FA5ED7EA8AC5E` |

## Statistical and methodological audit

The preregistered formal grid contains 1,152 complete trajectories and no
discarded runs.  It uses fixed gate thresholds rather than null-hypothesis
significance tests.  Pilot and formal seeds are disjoint.

Fallacy scan: **11/11 checked**.

- Simpson's paradox: mixing and delay strata retain the same correlation
  ordering; the inactive high-mixing baseline is shown separately.
- Ecological, Berkson, and collider fallacies: no cross-level inference,
  outcome-based sample selection, or regression controls are used.
- Base-rate neglect: the unknown kernel baseline is explicitly estimated and
  audited against its evaluation-only truth.
- Regression to the mean and survivorship bias: scenarios are fixed and all
  formal trajectories complete.
- Look-elsewhere effect and forking paths: the full grid and gates were frozen
  before fresh seeds; pilot results are excluded.
- Correlation/causation and reverse causality: conclusions are restricted to
  assigned parameters in the stated generative model.

## Mainline freeze

The CPU theory/controlled-experiment chain is now sufficient for a
theorem-first paper architecture:

1. minimax failure of linear speedup;
2. finite-gap affine delayed Markov convergence;
3. predictable mixing and correlation confidence sequences;
4. discrete latent collisions; and
5. continuous bounded-kernel latent sharing with unknown baseline.

Further scalar synthetic variants have declining scientific value.  The next
blocking evidence for an ICML submission is nonlinear multi-agent Markov
learning with learned representations and wall-clock/communication reporting.
That stage should be designed locally but is likely to require GPU execution.
