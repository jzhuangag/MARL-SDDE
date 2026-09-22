# CPU freshness oracle-headroom scan: frozen design

## Status and scientific role

This commit freezes an outcome-free feasibility scan.  It does not run the
grid, inspect a result, tune LSFF, evaluate an RL return, or authorize GPU work.
Its sole purpose is to decide whether variable conditional staleness risk has
enough intrinsic equal-cost scheduling value to justify a causal controller.

## Frozen conditional-risk process

Each 512-event path is a stationary two-state Markov chain.  Its high-state
prevalence is in `{0.1,0.25,0.5}` and its nontrivial transition eigenvalue is in
`{0,0.8,0.98}`.  Conditional stale-gradient MSE is one in the low state and is
multiplied by `{1,2,4,8}` in the high state.  Multiplier one is a stationary
negative control.  Arrival-fresh variance is `{0.5,1,2}` and exactly
`{10%,25%,50%}` of events may be refreshed.  The 64 seeds are 91001--91064.

The eventwise value of a fully charged refresh followed by optimal fusion is

```
R_k = A_k^2 / (A_k+B_k).
```

The noncausal oracle selects the exact number of largest `R_k`.  The strong
fixed comparator uses an evenly spaced schedule with the same exact count and
is allowed to choose its cyclic phase after seeing the realized path.  Thus any
reported oracle headroom is not due to a weak phase choice or unequal sensing
cost.

## Frozen mandatory gates

1. Every value is finite, and the oracle is never worse than the strong
   periodic comparator.
2. Multiplier-one stationary controls have at most `1e-10` relative headroom.
3. Across all dynamic rows, the geometric oracle/periodic risk ratio is at most
   `0.95`.
4. At least 60% of dynamic rows have at least 5% relative risk improvement.
5. For multiplier at least four, the geometric ratio is at most `0.90`.
6. In every persistence stratum, the dynamic geometric ratio is at most `0.97`.
7. Oracle and periodic schedules use the identical exact refresh count.
8. A clean rerun must reproduce `rows.csv` and `summary.json` byte for byte.

Failure of any gate stops the freshness mainline before a new MPE experiment.
Passing all gates authorizes only a causal LSFF CPU development study with new
seeds.  It does not authorize formal evidence, a standard benchmark, HPC4, or
GPU execution.

## Complexity and provenance

The scan contains 20,736 rows and uses FFT cyclic correlation for the
outcome-aware periodic phase.  It requires CPU only and writes no model or
trajectory.  The frozen JSON configuration, runner, algebra module, and tests
are committed together; the subsequent execution must record their hashes.
The frozen configuration SHA-256 is
`0476f5b9da3666712f0df2e913154330126300f271983debb769db46af7e9278`.
