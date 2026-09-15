# EXP-018B formal CPU validation

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-02
- Verification Status: FORMALLY VERIFIED WITH EXACT REPRODUCTION
- Preregistration Commit: `e4da74fc0240961ac90753c316067927f448c858`

## Decision

EXP-018B passes all eight frozen statistical/validity gates and the independent
exact-reproduction gate. A formal claim is authorized only for the registered
fixed-parameter nonlinear-TD gradient-variance calibration mechanism.

This result does not authorize an online participation controller, delayed or
dual-budget learning claims, nonlinear convergence, actor--critic claims, or a
GPU benchmark claim. EXP-018A remains an unchanged 5/7 pilot failure.

## Frozen-gate ledger

| Gate | Result | Frozen value |
|---|---:|---:|
| F1 shape/finite/unique | PASS | 18,432/18,432 finite unique rows; 192 seeds |
| F2 manifest/parameter freeze | PASS | manifest matched; all before/after parameter hashes equal |
| F3 q=1 CRN exactness | PASS | exact equality across rho in every registered stratum |
| F4 pairwise sharing | PASS | rho 0/0.5/0.9: 0 / 0.497538 / 0.899708 |
| F5 median equivalence | PASS | observed 0.077377; one-sided 97.5% upper 0.161906 <= 0.20 |
| F6 p90 equivalence | PASS | observed 0.139159; one-sided 97.5% upper 0.403933 <= 0.50 |
| F7 path-independent summary | PASS | input field is `projections.csv` |
| F8 scope boundary | PASS | fixed-gradient mechanism only |
| R1 exact reproduction | PASS | CSV, manifest, and summary byte-identical |

The two co-primary confidence bounds use 5,000 seed-cluster bootstrap
replications with frozen seed `18240101`. The 97.5% one-sided quantile for
each endpoint implements the registered Bonferroni familywise alpha 0.05.

The descriptive separated-direction diagnostic passes 45/48 paths (93.75%)
when adjacent theoretical variance factors differ by at least 5%. It was
preregistered as non-mandatory and is not used to rescue or redefine a gate.

## Exact-reproduction audit

| Artifact | SHA-256 | Bytes | Match |
|---|---|---:|---:|
| `projections.csv` | `0a955971530c20efc88cc5dae7ac3859107b36d3ddf75c5142cf3e36037b35bf` | 11,328,371 | exact |
| `static_manifest.json` | `0cfe24c37f4bba2895cc000f5ffd688977b3e618e55bbb9759ef7099cd8fcb25` | 5,041 | exact |
| `summary.json` | `37de678a35c685d00b03baf391ed9b079c6c84485acdaac1f4b03d05c667072d` | 1,761 | exact |

The formal run and reproduction used separate local result directories and
the same frozen runner/analyzer. Raw CSV artifacts remain in ignored local
directories. No HPC4, GPU, `/project`, or remote storage was used.

## Regression status

- EXP-018B targeted regression: `6 passed`;
- complete `experiments/nonlinear_markov_td` subtree: `90 passed, 7 skipped`;
- both committed JSON records: parse audit passed.

## Scientific interpretation

Across the frozen CartPole/Acrobot transition distributions, two regeneration
rates, two fixed ReLU-network initializations, three correlations, and four
participation levels, the variance of the averaged fixed-parameter stochastic
gradient is statistically calibrated by

`rho + (1-rho)/q`

at the registered aggregate median and p90 error tolerances. This establishes
the correlation--participation variance law in the deliberately narrow
nonlinear-gradient mechanism layer. It is a sound component of a future
theory-to-benchmark chain, not evidence that a learned adaptive controller
outperforms a strong fixed-q policy.
