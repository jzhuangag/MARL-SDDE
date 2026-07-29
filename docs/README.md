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
4. [`experiment_004_stagewise_controller.md`](experiment_004_stagewise_controller.md)
   records the predictable low-complexity joint step–participation experiment.
   The controller detected the dependence shift but did not improve MSE beyond
   dependence-aware step-size adaptation.
5. [`validation_exp004.md`](validation_exp004.md) records the paired-bootstrap
   interpretation, 11/11 statistical fallacy scan, and incomplete reproduction
   status.

## Code and canonical outputs

- Source and usage:
  `experiments/dependence_delay_linear/README.md`
- Registered baseline:
  `experiments/dependence_delay_linear/results/baseline/`
- Common-factor alignment sensitivity:
  `experiments/dependence_delay_linear/results/server_time_sensitivity/`
- Transient/stationary crossover:
  `experiments/dependence_delay_linear/results/crossover/`
- Stagewise controller:
  `experiments/dependence_delay_linear/results/stagewise/`

The `results/smoke/` directory is an implementation smoke test and must not be
used as scientific evidence. The same-seed reproduction directory is retained
locally but excluded from the public repository because it duplicates the
canonical outputs; its verification result is recorded in
`reproducibility_exp001.md`.

## Current decision

The project has strong evidence for correlation-limited speedup and
dependence-aware scalar step-size tuning. EXP-004 showed that a controller can
reduce its selected count from 32 to 4 after a common-noise shift, but this did
not improve MSE over retaining all agents and adapting only the step size.
Dynamic participation is therefore not currently supported as the main
accuracy contribution. It may remain useful as a resource-control mechanism,
which requires a communication-aware experiment.
