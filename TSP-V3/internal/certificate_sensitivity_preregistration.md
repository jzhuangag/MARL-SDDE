# TSP-V3-CERT-SENS-001 preregistration

## Purpose

This prospective CPU study tests two manuscript-facing claims without changing
the frozen TSP-V3 controller or using MARL evaluation return for selection:

1. predictable conservative upper certificates preserve empirical coverage
   with limited terminal-risk degradation;
2. the finite-horizon Lyapunov score improves over a one-step drift rule and a
   correlation-only participation rule under the same resource budget.

Only a complete development pass authorizes confirmation.  Only a complete
fresh-seed confirmation pass authorizes manuscript migration.

## Frozen design

- model: the existing four-state affine Markov learner;
- cells: persistence in \(\{0,0.9,0.98\}\), spatial correlation in
  \(\{0,0.9\}\), and maximum delay in \(\{0,8\}\), for 12 cells;
- catalogue: \(q\in\{1,4,16,32\}\), with the existing admissible spacings and
  optimized stable step sizes;
- resource budget: 128000 charged units;
- checkpoints: 41;
- development seeds: 33000001--33000008;
- confirmation seeds: 34000001--34000064;
- common random numbers are used across methods within each seed-cell pair.

The five frozen policies are:

- `finite_horizon`: minimize the complete finite-budget Lyapunov score;
- `conservative_moderate`: refit after multiplying \(K_q\) and \(\Omega_q\)
  by 1.10 and weakening the certified mixing log-rate by 0.05;
- `conservative_strong`: use factors 1.25 and log-rate weakening 0.10;
- `one_step_myopic`: minimize \(cR_0+d\) and ignore the remaining horizon;
- `correlation_only`: choose \(q\) from
  \((h+q)[\rho+(1-\rho)/q]\), then choose spacing and step size with the
  nominal certificate within that fixed \(q\).

The conservative levels are stress tests, not estimators fitted to outcomes.
The correlation-only rule deliberately uses spatial dependence and message
cost but omits mixing, delay, and the finite learning horizon.

## Frozen gates

Development requires all payloads finite, exact budget feasibility, all chosen
certificates stable, moderate and strong conservative terminal parameter-risk
ratios at most 1.10 and 1.25, respectively, and finite-horizon terminal-risk
ratios at most 0.95 versus myopic and 0.98 versus correlation-only.

Confirmation repeats those gates and additionally requires:

- all 24 level-by-cell one-sided 99 percent bootstrap upper means below their
  conservative certificate bounds;
- paired-bootstrap 95 percent upper ratios below one for both ablations.

Any failed mandatory development gate stops confirmation.  Any failed
confirmation gate keeps the study out of the manuscript.  Thresholds, cells,
policies, and seeds are not changed after observing outcomes.

## Reproducibility

The machine-readable registry is
`TSP-V3/experiments/certificate_sensitivity_preregistration.json`.
The runner records configuration, source, implementation, and Git hashes and
writes raw checkpoint metrics, action tables, seed-cell summaries, paired
comparisons, coverage results, and a decision summary.

