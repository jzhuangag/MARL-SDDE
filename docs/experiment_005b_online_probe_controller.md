# Experiment 005B: Online probe-charging participation controller

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-29
- Verification Status: UNVERIFIED
- Version Label: preregistration_v1

## Objective

Test whether a low-complexity online controller can use a charged observation
phase to select the participating-agent count and scalar step size under
clustered and heterogeneous Markov dependence, while approaching an
information oracle and improving over fixed-participation policies at the same
message-equivalent budget.

## Information and resource rules

- \(N=32\), with \(q\in\{1,2,4,8,16,32\}\).
- Every update costs \(q+4\) message-equivalent units.
- Total budget is 16,000 units.
- `adaptive_probe`, `all_agents_adaptive_eta`, and `probe_oracle` first pay for
  80 full-participation probe updates: \(80(32+4)=2880\) units.
- During the probe, the server may construct prefix aggregate residual streams
  because all 32 agents actually participate and are charged.
- After the probe, `adaptive_probe` observes only the selected \(q\) gradients.
  It receives no free information from non-participating agents.
- `fixed_q1_adaptive_eta` and `fixed_q8_adaptive_eta` observe and pay only for
  their respective agents during both estimation and exploitation.
- All policies stop before the next update would exceed the common budget.

## Markov dependence model

The observed noise of agent \(i\), assigned to one of four interleaved
clusters, is

\[
\xi_{i,t}
=\sqrt{\rho_g}a_i g_t
+\sqrt{\rho_c}b_i c_{\kappa(i),t}
+\sqrt{1-\rho_g-\rho_c}\,e_{i,t}.
\]

The global, cluster, and idiosyncratic factors are independent AR(1) processes
with coefficients 0.95, 0.70, and 0.20. The deterministic loadings \(a_i\) and
\(b_i\) vary between agents and are not ordered by latency. Four registered
environments are used:

| Name | \(\rho_g\) | \(\rho_c\) |
|---|---:|---:|
| independent | 0.0 | 0.0 |
| clustered | 0.0 | 0.6 |
| global | 0.6 | 0.0 |
| mixed | 0.3 | 0.4 |

Maximum deterministic delay is \(D\in\{4,16\}\). Agents are selected by the
fastest-prefix rule, while cluster labels and factor loadings are interleaved
with latency rank.

## Controller

Probe residual streams are converted to one scalar batch-means long-run
variance estimate for each candidate \(q\), using batch size 10. The controller
searches 17 scalar step sizes geometrically spaced from 0.0025 to 0.08. For
each \((q,\eta)\), a cached delayed linear transition computes predicted
finite-budget squared bias plus an estimated long-run-variance term. No
agent-by-agent covariance matrix or matrix preconditioner is formed.

`probe_oracle` has the same probe trajectory and cost but replaces estimated
long-run variances by the true values of the registered simulator. It is an
information benchmark, not an implementable policy.

## Policies

- `adaptive_probe`: estimated-LRV joint \(q,\eta\) controller;
- `probe_oracle`: true-LRV joint \(q,\eta\) controller with identical probe;
- `all_agents_adaptive_eta`: same probe, fixes \(q=32\), adapts only \(\eta\);
- `fixed_q8_adaptive_eta`: observes only eight agents and adapts \(\eta\);
- `fixed_q1_adaptive_eta`: observes only one agent and adapts \(\eta\).

## Experimental design

- 64 paired seeds, base seed 20260729;
- every seed is evaluated on all eight environment-delay cells and five
  policies;
- squared error is sampled on 101 common budget checkpoints;
- primary per-run metric: mean squared error over checkpoints 81--100;
- secondary metric: budget-normalized area under the squared-error curve;
- paired bootstrap uses 2,000 seed-level resamples, averaging registered cells
  within each seed before ratios are formed.

## Success gates

All gates are fixed before execution:

1. **independent participation:** median selected \(q\ge16\) for both
   independent cells;
2. **correlated response:** median selected \(q\le8\) in at least five of the
   six clustered/global/mixed cells;
3. **all-agent resource gain:** over the six correlated cells,
   `adaptive_probe / all_agents_adaptive_eta` final-window MSE ratio is at most
   0.80 and its paired-bootstrap 95% upper endpoint is below 0.95;
4. **oracle proximity:** over all eight cells,
   `adaptive_probe / probe_oracle` ratio is at most 1.25 and its 95% upper
   endpoint is below 1.40;
5. **fixed-policy adaptivity gain:** over all eight cells, adaptive-probe MSE
   is at most 0.95 times the best aggregate MSE among the three registered
   fixed-\(q\) policies;
6. **accounting and numerical validity:** all trajectories are finite, no
   policy exceeds budget, and the three full-probe policies each pay exactly
   2,880 probe units.

Failure of any gate remains a failed EXP-005B result. Parameters, policies, and
thresholds will not be changed after inspecting the output.

## Execution

- working directory: `experiments/dependence_delay_linear`;
- smoke command:
  `python run_online_participation.py --output-dir results/smoke/online_participation --num-seeds 4 --bootstrap-replications 100`;
- primary command:
  `python run_online_participation.py --output-dir results/online_participation --num-seeds 64 --bootstrap-replications 2000`;
- hard timeout: 15 minutes;
- environment: local Windows CPU; no GPU required.

## Expected outputs

- `per_seed_metrics.csv`;
- `actions.csv`;
- `budget_trajectories.csv`;
- `paired_bootstrap_ratios.csv`;
- `summary.json`;
- diagnostic figures for MSE and selected participation.

## Registered execution result

The 4-seed smoke run and 64-seed primary run completed within the registered
timeout. Sixteen deterministic tests passed. Five of the six pre-registered
gates passed, so the overall EXP-005B gate **failed**.

### Passed findings

- Independent cells selected median \(q=16\) for both \(D=4\) and \(D=16\).
- All six correlated cells selected median \(q\le4\):
  clustered \(q=4\), global \(q\in\{1,2\}\), and mixed \(q=4\).
- Over correlated cells, `adaptive_probe / all_agents_adaptive_eta` final-
  window MSE ratio was 0.3445 with paired-bootstrap 95% interval
  [0.2317, 0.5348].
- Over all cells, `adaptive_probe / probe_oracle` ratio was 1.0833 with
  interval [0.9375, 1.2699].
- Every trajectory was finite and within budget. The three registered full-
  probe policies paid exactly 2,880 units.

### Failed finding

The best registered fixed-participation policy was
`fixed_q1_adaptive_eta`. The `adaptive_probe / fixed_q1_adaptive_eta`
final-window MSE ratio was 1.0584 with interval [0.8210, 1.3573], failing the
registered ratio-at-most-0.95 criterion.

The mean all-cell budget-normalized area under the curve was 0.06425 for
`adaptive_probe` and 0.01732 for `fixed_q1_adaptive_eta`. The 80-round full
probe consumed 18% of the total budget and is therefore too expensive despite
producing sensible participation decisions.

## Reproduction and decision

An independent same-seed rerun produced byte-identical SHA-256 hashes for all
seven retained CSV, JSON, and figure artifacts. Reproducibility is
`REPRODUCIBLE`.

EXP-005B supports the learnability of the correlation-dependent participation
regime, but rejects the present full-probe implementation as the main
algorithm. A subsequent experiment may test sparse probing or an online
bandit-style controller, but the EXP-005B failure must remain unchanged.

