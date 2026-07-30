# Experiment 006B: Observable state-and-correlation participation control

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-30
- Verification Status: UNVERIFIED
- Version Label: preregistration_v1

## Objective

Test whether a low-complexity controller using only charged gradient probes can
exploit the state- and correlation-dependent oracle participation transitions
identified by EXP-006A.

This experiment does not use the true optimization error or simulator noise in
the adaptive policies. It estimates a scalar gradient-signal proxy and three
nonnegative covariance components from observed probe gradients. This closes
the observability gap in EXP-005C.

## Registered cells

- stationary dependence scenarios:
  - independent: \((\rho_g,\rho_c)=(0,0)\);
  - global: \((0.8,0)\);
  - clustered: \((0,0.8)\);
  - balanced: \((0.4,0.4)\);
- maximum delay \(D\in\{4,16\}\);
- initial error 0.3;
- eight blocks of message-equivalent budget 2,000;
- total budget 16,000 and update overhead 4;
- 64 paired seeds beginning at 20260730;
- 81 common budget checkpoints.

The high-dependence and error/budget settings lie inside or adjacent to the
actionable rectangles registered and discovered in EXP-006A. No cell is chosen
from EXP-006B outcomes.

## Observable controller

At the beginning of every block, each adaptive policy performs eight charged
updates with the fastest eight agents and step size 0.02. Probe cost is

\[
8\text{ blocks}\times 8\text{ probes}\times(8+4)=768,
\]

or 4.8% of total budget.

The controller retains at most 32 probe-gradient vectors. It centers each
agent's gradients over the rolling window and fits the same three-component
nonnegative moment model used in EXP-005C. The scalar state proxy is the
absolute rolling mean probe gradient divided by curvature, clipped to
\([0.01,1]\). It then searches the registered six agent counts and 17 scalar
step sizes using a constant delayed-history proxy at that observable state
level. Complexity is \(O(Wq_p+|\mathcal Q||\mathcal H|)\) scalar operations per
block, with \(W=32\), probe width \(q_p=8\), six agent counts, and 17 step
sizes. No covariance matrix, neural controller, actor--critic, or
preconditioner is used.

## Policies

- `state_correlation_adaptive`: estimated dependence plus observable state
  proxy;
- `correlation_only_adaptive`: the same probes and dependence estimator, but
  fixed state proxy 0.3;
- `state_oracle`: true dependence and true current delayed error history,
  without probe cost;
- `fixed_q1_oracle_eta`;
- `fixed_q4_oracle_eta`;
- `fixed_q8_oracle_eta`;
- `fixed_q32_oracle_eta`.

The fixed policies know the true dependence and select their best registered
step size each block. They are stronger than implementable fixed baselines.

## Primary metric

For every seed, delay, scenario, and policy, compute the mean squared error over
all nonzero budget checkpoints and divide it by the same-cell
`state_oracle` value. The primary dynamic score is the arithmetic mean of these
normalized cell scores. The hindsight-best fixed policy is chosen only after
averaging its preregistered scores.

Paired bootstrap resampling uses seeds as the resampling unit and retains every
scenario and both delays. Two thousand bootstrap replications are used.

## Action agreement

For blocks 1--7, compare the adaptive and state-oracle selected counts in every
paired seed/delay/scenario cell. An action agrees when the counts differ by at
most a factor of two. Block 0 is excluded as the registered warm-up block.

## Go/no-go gates

All gates must pass:

1. **state value**:
   `state_correlation_adaptive / correlation_only_adaptive` is at most 0.90
   and its paired-bootstrap 95% upper endpoint is below 1;
2. **best-fixed improvement**:
   `state_correlation_adaptive / hindsight-best-fixed` is at most 0.90 and its
   95% upper endpoint is below 1;
3. **oracle proximity**:
   the adaptive normalized score is at most 1.50 and its 95% upper endpoint is
   below 1.75;
4. **action agreement**:
   at least 50% of registered post-warm-up actions agree with the oracle
   within a factor of two;
5. **probe budget**:
   every adaptive run spends exactly 768 probe units and no more than 5% of
   total budget;
6. **accounting and numerical validity**:
   every run and metric is finite, every update is charged, all 64 seeds and
   eight cells complete, and no policy exceeds block or total budget.

Failure of any gate rejects this observable controller as the paper's main
algorithm. Results will not be repaired by changing the proxy clipping,
rolling window, probe schedule, cell weights, baselines, or thresholds.

## Execution

- working directory: `experiments/dependence_delay_linear`;
- smoke:
  `python run_state_correlation.py --output-dir results/smoke/state_correlation --num-seeds 4 --bootstrap-replications 100`;
- primary:
  `python run_state_correlation.py --output-dir results/state_correlation --num-seeds 64 --bootstrap-replications 2000`;
- local Windows CPU; no GPU;
- hard timeout: 20 minutes.

## Expected outputs

- `per_seed_cell_metrics.csv`;
- `block_actions.csv`;
- `run_accounting.csv`;
- `budget_trajectories.csv`;
- `paired_bootstrap_ratios.csv`;
- `summary.json`;
- MSE, participation, and state-proxy figures.

## Execution outcome

The registered 64-seed run completed in 189.9 seconds and failed four
scientific gates. State-aware/correlation-only was 1.131, adaptive/best-fixed
was 1.728, adaptive/oracle was 2.652, and action agreement was 34.35%. Probe
budget and accounting/numerical validity passed. The overall verdict is
**FAIL**.

A deterministic full rerun completed in 129.6 seconds and all nine artifacts
matched byte-for-byte. The root-cause audit found accurate dependence-component
estimates but a poorly calibrated gradient-magnitude state proxy: its
log-correlation with true error was 0.326 and it overestimated true error by a
median factor of 2.89. See `validation_exp006b.md`.
