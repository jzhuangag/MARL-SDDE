# T-018 corrected static scan results

This file records the T-018 bookkeeping/statistical erratum. The original
`t018_static_scan_results.*` files remain immutable provenance. No trajectory,
pilot, formal run, HPC4 job, GPU job, or scientific outcome was generated.

## Corrected accounting

- Grid hash: `c5d2dd5ddac7540888d708ab59d4e3954994da018951797a79d05200ef0ee2db`
- All scenarios: 3456
- Finite-`B_value` scenarios: 3448
- Search-censored scenarios: 8
- Censored fraction: `0.0023148148148148147`
- Analytic cells after censor correction: 69024
- Finite active-zone coverage: `1`
- Finite `Z` cells: 27584
- Practical-effect finite `Z` cells: 19337
- Descriptive practical-effect coverage among finite `Z` cells:
  `0.70102233178654294`
- Message-binding finite scenarios: 1766
- Environment-binding finite scenarios: 1402

The practical-effect coverage is descriptive. Commit 1 froze a 3% per-cell
practical-effect threshold, not a retrospective effect-coverage pass gate.

## Robust finite `Z` width summary

```json
{
  "count": 3448,
  "iqr_high_q75": 156.0,
  "iqr_low_q25": 51.0,
  "maximum": 749.0,
  "median": 88.0,
  "p90": 238.0,
  "p99": 478.58999999999924
}
```

## Novelty status

- N1 safety aligned: `true`
- N2 broad-grid finite `Z >= 25%`:
  `true`
- N3 practical effect recorded descriptively:
  `true`
- N4 delay or dual-budget directional effect:
  `true`
- N5 no information-only leakage:
  `true`
- N6 active subsets outcome-free:
  `true`
- N7 both binding mechanisms present:
  `true`

Final corrected decision: **A**.
