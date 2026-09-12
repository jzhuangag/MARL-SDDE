# T-018 static scan preregistration

## Scope

T-018 freezes an outcome-free analytic scan for learning-value separation,
safety-metric alignment, and mechanism activity. It does not revive EXP-016A
and does not authorize any pilot, formal run, HPC4 job, GPU job, trajectory,
or scientific outcome.

The scan is separated into two commits:

1. freeze this grid, formulas, active-zone definition, practical thresholds,
   and no-outcome statement;
2. execute only the analytic/static scan and record all cells.

## Frozen history

- EXP-015A remains a 7/8 failure with gate `0.80` and observed `0.777778`.
- EXP-016A original preregistration remains immutable provenance.
- EXP-016A Amendment 1 remains immutable provenance.
- Original EXP-016A hash:
  `bb3ab51bc64d4ee334e7c5da6b6e7a4e7ffd303692abb6a5e48d06e48bb9baf5`.
- Amendment 1 hash:
  `a6312a4769457c3d73aedea60f9c3523a2860d6fc0499ad1ccdb6124188412d0`.
- G6 per-cell feasibility failure and G8 active subset `0` are not
  reinterpreted.

If T-018 succeeds, it can only authorize a separately preregistered EXP-016B.
It cannot modify or run EXP-016A.

## Frozen grid

The machine-readable manifest is `t018_scan_manifest.json`.

- Grid hash:
  `c5d2dd5ddac7540888d708ab59d4e3954994da018951797a79d05200ef0ee2db`
- Scenario count: 3456
- Budget points per scenario: 10
- `Q in {8,16,32}`
- `theta_low = 0.05`
- `theta_high in {0.5,1.0,2.0,4.0}`
- `lambda in {0.2,0.7,0.9,0.94}`
- `D in {0,4,12,24}`
- `h in {4,16,32}`
- budget rays: balanced, message-limited, environment-limited
- `epsilon_safe in {0.10,0.20}`
- finite action catalogue: `q in {2,4,8,16,32}`, `b in {1,2,4,8}`,
  `eta=1.0`

For each scenario the ten registered budget labels are:
`half_BN`, `near_BN`, `at_BN`, `at_Bid`, `mid_id_value`, `near_value`,
`last_Z_integer`, `at_value`, `above_BS`, and `double_BS`.

## Definitions

`B_N` is the necessary identification scale from the directional KL
lower-bound samples. `B_id` is the first scale at which the information-only
baseline can afford a statistically reliable fixed probe. `B_value` is the
first scale at which the learning-aware rule can amortize the probe and pass
the theorem-facing safety rule.

The active separation zone is

```text
Z = {B : B_id <= B < B_value}.
```

Inside `Z`, the information-only rule probes, while the learning-aware rule
falls back because identification has not yet become downstream worthwhile
and safe.

## Frozen gates

- N1: safety theorem and experiment-facing metric are aligned.
- N2: at least 25% of nondegenerate broad-grid scenarios have nonempty `Z`.
- N3: a preregistered fraction of `Z` cells has at least 3% analytic
  downstream regret increase from premature information-only probing.
- N4: delay or dual-budget structure has a nontrivial directional effect on
  `Z` width.
- N5: the information-only baseline has no learning-risk taint.
- N6: active subsets are generated from frozen analytic rules only.
- N7: both message-binding and environment-binding mechanism cells exist.

The `25%` and `3%` thresholds may not be weakened after the scan.

## No-outcome statement

This preregistration freezes only analytic formulas and a deterministic scan
grid. It contains no trajectory, pilot, formal, HPC4, GPU, or scientific
outcome.
