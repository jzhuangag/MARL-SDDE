# Experiment 006C: Lyapunov-surrogate participation control

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-30
- Verification Status: UNVERIFIED
- Version Label: preregistration_v1

## Decision motivating the experiment

EXP-006B rejected the rolling gradient-magnitude state proxy. The dependence
estimator separated independent, global, clustered, and balanced noise, but
persistent common Markov noise made the absolute block-mean gradient a biased
proxy for optimization error. EXP-006C therefore changes the state estimator,
not the registered participation hypothesis.

The new controller maintains one scalar model-based Lyapunov surrogate
\(U_k\) for the squared optimization error. It uses no true error, simulator
state, or uncharged observation. The controller remains a finite lookup over
six participation counts and 17 scalar step sizes; it introduces no learned
policy, matrix preconditioner, or actor--critic.

## Registered cells and independence

- stationary dependence scenarios:
  - independent: \((\rho_g,\rho_c)=(0,0)\);
  - global: \((0.8,0)\);
  - clustered: \((0,0.8)\);
  - balanced: \((0.4,0.4)\);
- maximum delay \(D\in\{4,16\}\);
- initial error 0.3;
- eight blocks of message-equivalent budget 2,000;
- total budget 16,000 and update overhead 4;
- 64 paired seeds beginning at **20260830**, disjoint from EXP-006B;
- 81 common budget checkpoints;
- 2,000 seed-level paired bootstrap replications.

The cells, metrics, and gates are frozen before any EXP-006C smoke or primary
run. EXP-006B results determine only the replacement of the failed state proxy.

## Observable Lyapunov surrogate

Every implementable adaptive policy performs eight charged \(q=8\) probe
updates with step size 0.02 at the start of each block. The rolling 32 probe
vectors estimate three nonnegative long-run-variance components after
agent-wise temporal centering.

For `lyapunov_state_adaptive`, initialize

\[
U_0 = x_0^2=0.09.
\]

After the probes, propagate a constant delayed-history vector
\(\sqrt{U_k}\mathbf 1\) through the registered stable delayed linear model
using the estimated aggregate long-run variance. The resulting predicted risk
is \(U_k^{\rm probe}\). Choose \((q_k,\eta_k)\) minimizing the registered
finite-budget risk over the remaining block, and set

\[
U_{k+1} =
\mathcal R(q_k,\eta_k,U_k^{\rm probe},
           \widehat{\Omega}_k,D).
\]

This is an observable one-dimensional recursion. The true error is recorded
only for audit plots and is never read by the controller.

Per-block decision complexity is
\(O(Wq_p+|\mathcal Q||\mathcal H|)\), with \(W=32\), \(q_p=8\),
\(|\mathcal Q|=6\), and \(|\mathcal H|=17\). Memory is \(O(Wq_p+D)\).

## Policies

- `lyapunov_state_adaptive`: estimated dependence and recursively propagated
  scalar Lyapunov surrogate;
- `raw_state_adaptive`: the preregistered EXP-006B gradient-magnitude proxy,
  rerun on the fresh seeds;
- `correlation_only_adaptive`: estimated dependence and fixed state proxy 0.3;
- `state_oracle`: true dependence and true current delayed error history,
  without probe cost;
- `fixed_q1_oracle_eta`;
- `fixed_q4_oracle_eta`;
- `fixed_q8_oracle_eta`;
- `fixed_q32_oracle_eta`.

Fixed policies know the true dependence and choose their best registered
step size in each block, so they remain deliberately strong baselines.

## Primary metric and action agreement

For every seed, delay, scenario, and policy, average squared error over all
nonzero budget checkpoints and divide by the paired `state_oracle` value.
The primary score is the arithmetic mean of these normalized cell scores.
The hindsight-best fixed baseline is selected only among the four registered
fixed policies after their scores are averaged.

For blocks 1--7, an adaptive action agrees with the oracle if their selected
agent counts differ by at most a factor of two. Block 0 is the registered
warm-up block and is excluded.

## Go/no-go gates

All gates must pass:

1. **replacement value**:
   `lyapunov_state_adaptive / raw_state_adaptive` is at most 0.90 and its
   paired-bootstrap 95% upper endpoint is below 1;
2. **state value**:
   `lyapunov_state_adaptive / correlation_only_adaptive` is at most 0.90 and
   its 95% upper endpoint is below 1;
3. **best-fixed improvement**:
   `lyapunov_state_adaptive / hindsight-best-fixed` is at most 0.90 and its
   95% upper endpoint is below 1;
4. **oracle proximity**:
   the Lyapunov controller's normalized score is at most 1.50 and its 95%
   upper endpoint is below 1.75;
5. **action agreement**:
   at least 50% of registered post-warm-up actions agree with the oracle
   within a factor of two;
6. **probe budget**:
   every implementable adaptive run spends exactly 768 probe units and no
   more than 5% of total budget;
7. **accounting and numerical validity**:
   all \(64\times4\times2\times8=4096\) runs and metrics are finite, every
   update is charged, and no run exceeds block or total budget.

Failure of any gate rejects this exact Lyapunov-surrogate controller as the
paper's main algorithm. Results will not be repaired by changing initialization,
surrogate propagation, clipping, probe schedule, cell weights, baselines, or
thresholds.

## Execution

- working directory: `experiments/dependence_delay_linear`;
- smoke:
  `python run_lyapunov_state.py --output-dir results/smoke/lyapunov_state --num-seeds 4 --bootstrap-replications 100`;
- primary:
  `python run_lyapunov_state.py --output-dir results/lyapunov_state --num-seeds 64 --bootstrap-replications 2000`;
- exact reproduction:
  repeat the primary command with
  `--output-dir results/reproduction/lyapunov_state`;
- local Windows CPU; no GPU;
- hard timeout: 20 minutes per run.

## Expected outputs

- `per_seed_cell_metrics.csv`;
- `block_actions.csv`;
- `run_accounting.csv`;
- `budget_trajectories.csv`;
- `paired_bootstrap_ratios.csv`;
- `summary.json`;
- MSE, participation, and surrogate-calibration figures.
