# Causal LSFF conditional-risk confirmation

## Decision

All eight frozen gates pass on 64 new seeds.  The causal multi-resource
Lyapunov mechanism robustly captures the conditional-risk scheduling
opportunity found by the noncausal oracle.  This authorizes an independently
designed arrival-fresh MPE CPU experiment; it is not RL-return evidence and
does not authorize GPU or formal runs.

## Provenance and reproduction

- Preregistration commit: `9432c2a`
- Configuration SHA-256:
  `b8e7e3d82d93c48e1e190b3f2da841858e4ac8e730d8b5f587bd29fd20227f3c`
- Seeds: 92001--92064, disjoint from development seeds 91001--91064
- Rows: 20,736
- Primary and reproduction runtime: 11.14 s each on local CPU
- `rows.csv` SHA-256 in both runs:
  `9c86a10c841c49a8fd4603cbcc5748c912ab1c8b5d557ee702e500725320581a`
- `summary.json` SHA-256 in both runs:
  `169f34576678f40ac853db4a4638c9042fa92965390602e14f34d6c127ba4acf`

## Frozen gate results

| Gate | Requirement | Observed | Status |
|---|---:|---:|---:|
| C1 | finite and correctly ordered | all rows | pass |
| C2 | stationary ratio error at most `1e-10` | `3.11e-15` | pass |
| C3 | dynamic geometric ratio at most `.90` | `.849767` | pass |
| C4 | better in at least 85% dynamic rows | `95.422%` | pass |
| C5 | median oracle capture at least 70% | `83.871%` | pass |
| C6 | each persistence ratio at most `.95` | `.796635/.824553/.934161` | pass |
| C7 | mean utilization at least 99%; no cap violation | `100%`; none | pass |
| C8 | byte-exact isolated rerun | both artifacts | pass |

The development geometric ratio was `.850166`; the new-seed value is
`.849767`.  The development median oracle capture was 85.0%; confirmation gives
83.87%.  The small shift and identical direction across persistence strata
show that the mechanism result did not depend on the development paths.

## Scope boundary and next experiment

The confirmation supplies the middle link of the proposed ICML story:
time-varying freshness has intrinsic equal-cost value, and a causal O(1)
Lyapunov queue captures most of it under hard finite-horizon budget caps.  It
does not yet establish the first and last links: that a real asynchronous MARL
learner can form conservative `A_k,B_k`, and that lower certified gradient MSE
translates into better return after fully charging fresh rollouts and service
time.

The next CPU design must therefore collect each birth packet at its launch
policy, optionally collect an independent gradient at the actual completion
policy, fuse them with the frozen formula, and compare against never refresh,
always refresh, and strong fixed-period refresh under equal total actor
transitions or an explicit return--resource Pareto analysis.  It must log true
arrival-gradient discrepancy to evaluate coverage without exposing that value
to the online action.  Markov bias, KL reference-state alignment, and refresh
wall-clock service remain theorem gates.

No GPU/HPC4 work is justified yet.
