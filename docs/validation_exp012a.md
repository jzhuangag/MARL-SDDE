# EXP-012A validation: latent collision certificate

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Mode: validate
- Verification Status: VERIFIED
- Formal execution: 128 fresh seeds in each of nine scenarios
- Reproducibility verdict: REPRODUCIBLE

## Decision

**PASS.**  Both validity gates and all four scientific gates pass.  The
certificate no longer observes hidden sharing masks; it uses only equality of
two agent Markov samples and the known independent collision baseline.

## Formal results

| Gate | Threshold | Result |
|---|---:|---:|
| Joint anytime coverage | at least 97.5% | 100% |
| Conditional action safety | all updating radii \(<1\) | pass; maximum 0.993830 |
| Correlation ordering | non-decreasing \(\rho^+\) | pass at all persistences |
| Fast-mixing identification | fixed three-part gate | pass: 0.169, 0.670, 1.000 |
| Persistent probe nonvacuity | at least 50 probes | pass; minimum median 82 |
| Participation response | weak response everywhere, one strict | pass |

At \(p=.5\), median certified participation for \(D=0\) changes from
\(q=8\) at \(\rho=0\), to \(q=2\) at \(\rho=.5\), and to \(q=1\) at
\(\rho=.9\).  At \(D=2\), every cell selects \(q=1\); delay has already
exhausted the participation control margin.

The inverse-mixing-gap penalty remains visible.  At \(p=.98,\rho=0\), the
median latent-correlation upper bound is 0.609 despite 83 median probes.  The
certificate remains safe but cannot be described as uniformly sharp near
unit persistence.

## Reproducibility

A clean same-seed rerun reproduced all four core artifacts byte-for-byte:

| Artifact | SHA-256 |
|---|---|
| `latent_collision_summary.png` | `2FA102E76CFA94FA4A48234BC36202C6FFD230413F815E8065631754C0968026` |
| `metrics.csv` | `AC0783ED4D2ECD4821E82030C10D83E86B8C64C5CFA1D2B0170C2CA29597A028` |
| `summary.json` | `62379EFFFC3AB0838E51DAA39FA1272EFD866919D875F053C4595574B94B5795` |
| `traces.csv` | `0C6B1C700FABA3BE071A30893CD2D727190F5608A1D26D64C98D3997E58658C5` |

## Statistical and methodological audit

The experiment evaluates deterministic preregistered gates over simulated
trajectories; it does not use null-hypothesis \(p\)-values.  All 1,152 formal
certificate trajectories completed, so there is no attrition.  The pilot and
formal seeds are disjoint.

Fallacy scan: **11/11 checked**.

| Fallacy | Finding |
|---|---|
| Simpson's paradox | No reversal; persistence and delay are reported separately. |
| Ecological fallacy | No individual-agent inference is drawn from scenario aggregates. |
| Berkson's paradox | Full preregistered grid is retained; no outcome-based selection. |
| Collider bias | No post-treatment control or regression adjustment is used. |
| Base-rate neglect | The natural collision baseline \(c_\pi=.5\) is explicit. |
| Regression to the mean | Scenarios are fixed, not selected by extreme pilot outcomes. |
| Survivorship bias | Zero missing or discarded formal trajectories. |
| Look-elsewhere effect | Gates and scenario grid were frozen before fresh seeds. |
| Garden of forking paths | Pilot tuning is disclosed and excluded from evidence. |
| Correlation versus causation | Claims are confined to the specified generative model. |
| Reverse causality | \(p\) and \(\rho\) are assigned data-generating parameters. |

## Mainline decision

The observable-sharing limitation of EXP-011B is closed for discrete
pair-sharing Markov data.  The next CPU generalization should replace exact
equality by a bounded similarity kernel and estimate or lower-bound its
independent-source baseline.  Arbitrary Gaussian common factors and nonlinear
MARL remain outside the current certificate.
