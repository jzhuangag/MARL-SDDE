# T-018 static scan results

This file records an outcome-free analytic scan. No trajectory, pilot,
formal run, HPC4 job, GPU job, or scientific outcome was generated.

## Summary

- Grid hash: `c5d2dd5ddac7540888d708ab59d4e3954994da018951797a79d05200ef0ee2db`
- Scenarios scanned: 3456
- Registered cells scanned: 69120
- Nondegenerate scenarios: 3448
- Scenarios with nonempty `Z`: 3448
- Active-zone coverage: `1`
- `Z` cells: 27648
- Practical-effect `Z` cells: 19353
- Effect coverage among `Z` cells: `0.69997829861111116`
- Message-binding scenarios: 1772
- Environment-binding scenarios: 1404

## Novelty gates

- N1 safety aligned: `true`
- N2 broad-grid `Z >= 25%`: `true`
- N3 practical effect present: `true`
- N4 delay or dual-budget directional effect:
  `true`
- N5 no information-only leakage:
  `true`
- N6 active subsets outcome-free:
  `true`
- N7 both binding mechanisms present:
  `true`

## Monotonic scan summaries

Mean `Z` width by delay:

```json
{
  "0": 4737.630787037037,
  "12": 4749.709490740741,
  "24": 4761.260416666667,
  "4": 4741.670138888889
}
```

Mean `Z` width by overhead:

```json
{
  "16": 118.39756944444444,
  "32": 14002.113715277777,
  "4": 122.19184027777777
}
```

Mean `Z` width by budget ray:

```json
{
  "balanced": 93.04600694444444,
  "environment_limited": 14004.563368055555,
  "message_limited": 145.09375
}
```

## Decision

Final T-018 decision: **A**.

Decision A authorizes only a future, separately preregistered EXP-016B design
stage. It does not authorize running EXP-016A or any pilot in this commit.
