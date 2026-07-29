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
6. [`experiment_005a_budget_participation_surface.md`](experiment_005a_budget_participation_surface.md)
   records the pre-registered resource-matched participation surface. Its five
   gates passed: independent-noise cells selected all 32 agents, whereas the
   high-correlation cells selected one agent under the primary message budget.
7. [`validation_exp005a.md`](validation_exp005a.md) records the numerical
   validation, 11/11 fallacy scan, scope warnings, and byte-identical rerun.
8. [`experiment_005b_online_probe_controller.md`](experiment_005b_online_probe_controller.md)
   records the charged online controller. It learned the correct participation
   direction and beat all-agent control, but failed to beat fixed \(q=1\).
9. [`validation_exp005b.md`](validation_exp005b.md) records the paired-bootstrap
   audit, failed overall gate, 11/11 fallacy scan, and exact reproduction.
10. [`experiment_005c_sparse_dynamic_controller.md`](experiment_005c_sparse_dynamic_controller.md)
    records the final sparse, nonstationary go/no-go design, its initial
    timeout, authorized execution v2, and failed registered decision.
11. [`validation_exp005c_timeout.md`](validation_exp005c_timeout.md) preserves
    the initial `CANNOT_VERIFY` audit.
12. [`validation_exp005c.md`](validation_exp005c.md) records the completed
    64-seed result, exact reproduction, oracle-gate mismatch, and 11/11 fallacy
    scan.

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
- Budget-matched participation:
  `experiments/dependence_delay_linear/results/budget_participation/`
- Online probe-charging controller:
  `experiments/dependence_delay_linear/results/online_participation/`

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
EXP-005A subsequently confirmed that participation can be a strong
resource-control mechanism: the resource-matched optimum changes from all
agents under independent noise to one agent under strong common noise. The
next experiment tested an online controller that observes only participating
agents and charges every exploration probe. It correctly reduced participation
under correlated noise but its 18% full-probe cost prevented it from beating
the best fixed-\(q\) policy. EXP-005C reduced exploration to 2.4% and introduced
within-run regime shifts. Its authorized optimized execution was exactly
reproduced but failed three scientific gates. The audit also showed that the
piecewise oracle retained median \(q=32\) in every regime, so the current test
rejects the controller without resolving the broader participation hypothesis.
