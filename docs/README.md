# SDDE research experiment index

## Active direction

The current feasibility question is whether persistent cross-agent dependence
and heterogeneous delay create a useful participation-control problem beyond
the standard linear-speedup analysis.

## Experiment records

1. [`experiment_001_dependence_delay_go_nogo.md`](experiment_001_dependence_delay_go_nogo.md)
   records the registered baseline, the exact linear model, the four decision
   gates, EXP-001 results, and the EXP-002 common-factor alignment sensitivity.
2. [`experiment_003_transient_stationary_crossover.md`](experiment_003_transient_stationary_crossover.md)
   records the post-baseline finding that the best fixed-step participation
   level changes between transient and stationary regimes.
3. [`reproducibility_exp001.md`](reproducibility_exp001.md) records analytic
   implementation checks, Monte Carlo agreement, and the independent rerun.

## Code and canonical outputs

- Source and usage:
  `experiments/dependence_delay_linear/README.md`
- Registered baseline:
  `experiments/dependence_delay_linear/results/baseline/`
- Common-factor alignment sensitivity:
  `experiments/dependence_delay_linear/results/server_time_sensitivity/`
- Transient/stationary crossover:
  `experiments/dependence_delay_linear/results/crossover/`

The `results/smoke/` directory is an implementation smoke test and must not be
used as scientific evidence. The same-seed reproduction directory is retained
locally but excluded from the public repository because it duplicates the
canonical outputs; its verification result is recorded in
`reproducibility_exp001.md`.

## Current decision

The project has strong evidence for correlation-limited speedup and for
dependence-aware step-size tuning. It does not yet have evidence that the
jointly tuned 500-step optimum rejects agents. The stronger, currently
supported mechanism is stage-dependent participation at a fixed constant step:
more agents help the early transient, while fewer agents lower the stationary
error floor.
