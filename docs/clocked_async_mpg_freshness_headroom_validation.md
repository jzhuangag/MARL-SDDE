# CPU freshness oracle-headroom validation

## Decision

All eight frozen feasibility gates pass.  Variable conditional staleness risk
has material equal-cost scheduling value, so the next step may implement and
test a causal Lyapunov-Scheduled Freshness Fusion (LSFF) controller on CPU.  The
scan is not an RL outcome, does not validate the observable certificate, and
does not authorize formal or GPU experiments.

## Frozen execution

- Preregistration commit: `34956ee6d75abcf52effb01fdc6bc1f46c4521da`
- Configuration SHA-256:
  `0476f5b9da3666712f0df2e913154330126300f271983debb769db46af7e9278`
- Rows: 20,736, of which 15,552 are dynamic and 5,184 are stationary controls
- Primary runtime: 3.71 s on local CPU
- Clean reproduction runtime: 3.85 s on local CPU
- `rows.csv` SHA-256 in both runs:
  `2c9a11840620546c72d3bcf35fcc11c07c29034a77d4dba5b60fd4914e34bed4`
- `summary.json` SHA-256 in both runs:
  `bbafcc43847e82581ab04b95ff772fb4af18e2b76bec3885b0c164824c0dd442`

The strong periodic comparator uses the same exact number of refreshes and is
allowed to select its best cyclic phase after seeing the path.  The oracle is
also outcome-aware and selects the largest eventwise fusion values.  The
comparison therefore measures an optimistic scheduling ceiling, not a causal
algorithm.

## Results and frozen gates

| Gate | Frozen requirement | Result | Status |
|---|---:|---:|---:|
| G1 | finite; oracle never worse | all 20,736 rows | pass |
| G2 | stationary improvement at most `1e-10` | `0` | pass |
| G3 | dynamic geometric risk ratio at most `.95` | `.800487` | pass |
| G4 | at least 60% dynamic rows improve by 5% | `78.543%` | pass |
| G5 | high-contrast geometric ratio at most `.90` | `.745105` | pass |
| G6 | every persistence ratio at most `.97` | `.780884/.783117/.838782` | pass |
| G7 | identical exact refresh count | all rows | pass |
| G8 | byte-exact clean reproduction | both artifacts identical | pass |

The dynamic mean reduction in total conditional MSE relative to the
best-phase periodic schedule is 18.10%.  Markov persistence reduces but does
not remove the opportunity: even at persistence `.98`, the geometric risk
ratio is `.838782`.  The stationary negative control has exactly zero
headroom, as required.  These facts support the mechanism's phase premise:
state-dependent sensing matters only when the value of freshness changes over
time.

## Statistical and methodological validation

No p-value, confidence interval, or population-level inferential claim is made.
This is an exact functional comparison over a frozen finite grid, so the
reported effect is practical headroom on that grid rather than a sampling
significance statement.

The required 11/11 fallacy scan found no contradiction within this scope:

1. Simpson's paradox: aggregate direction remains positive in every
   persistence stratum; other strata remain available in the raw rows.
2. Ecological fallacy: no per-agent or RL-return inference is drawn from grid
   aggregates.
3. Berkson's paradox: the full frozen grid is retained; no passing subset was
   selected.
4. Collider bias: no regression or conditioned causal estimate is used.
5. Base-rate neglect: high-state prevalences are explicit grid dimensions.
6. Regression to the mean: no extreme-outcome selection or pre/post estimate
   is used.
7. Survivorship bias: all 20,736 planned rows are finite and included.
8. Look-elsewhere effect: gates and thresholds were committed before execution;
   all gates are reported.
9. Garden of forking paths: configuration, code, gates, and seeds were frozen
   in the preregistration commit.
10. Correlation versus causation: the scan claims conditional-risk headroom,
    not improved RL return.
11. Reverse causality: Markov regimes generate the stipulated risk path; no
    empirical directional association is interpreted causally.

## What this permits and what remains

The CPU oracle-headroom gate is now closed positively.  The next experiment
must be causal and observable: it must estimate birth bias and fresh variance,
charge every arrival-time rollout, compare with never/always/best fixed-period
refresh under equal actor-transition or wall-clock budgets, and report what
fraction of the oracle frontier LSFF captures on fresh seeds.

The performance-bound gate remains incomplete.  In particular, the
cross-policy KL certificate must match the policy-gradient reference-state
distribution; Markov estimator bias must be explicit; queue cost optimality
needs an exogeneity/ergodicity or controlled-state theorem; and refresh service
time must be composed with the existing wall-clock result.  No GPU benchmark
is authorized until both that theory gate and a causal MPE CPU gate pass.
