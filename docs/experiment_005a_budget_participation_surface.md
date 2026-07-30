# Experiment 005A: Budget-matched participation surface

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-29
- Verification Status: UNVERIFIED
- Version Label: preregistration_v1

## Objective

Test the necessary mechanism behind an agent-participation contribution:
whether the risk-minimizing number of participating agents changes with
cross-agent correlation and delay once communication or wall-clock resources
are matched.

This experiment is a falsification gate. It does not test an online controller.
EXP-005B will be attempted only if a nontrivial participation surface exists.

## Model and fixed design

The scalar delayed linear Markov model and exact augmented-state risk
calculation in `linear_model.py` are used without modifying their dynamics.
For every candidate action, only the selected agents enter the aggregate and
only their messages are charged.

- maximum available agents: \(N=32\);
- candidates: \(q\in\{1,2,4,8,16,32\}\);
- step-size grid: 17 geometrically spaced values from 0.0025 to 0.08;
- correlations: \(\rho\in\{0,0.3,0.6,0.9\}\);
- maximum delay: \(D\in\{4,16\}\);
- alignments: server time (primary) and sample time (sensitivity);
- selection rules:
  - `fastest`: the fastest \(q\) agents;
  - `uniform_rank`: deterministic latency-rank coverage, used to check that a
    result is not solely caused by deleting stragglers.

For each cell and resource definition, the finite-horizon mean-square error
(MSE) is minimized jointly over the pre-specified \(q\) and step-size grids.
Unstable actions remain in the raw surface but cannot be selected.

## Resource definitions

### Message-equivalent budget (primary)

Each server update costs

\[
c_{\mathrm{msg}}(q)=q+c_0,
\]

where the primary fixed per-update overhead is \(c_0=4\), the total budget is
\(B=6400\), and the update horizon is
\(\lfloor B/c_{\mathrm{msg}}(q)\rfloor\).
Sensitivity overheads are \(c_0\in\{0,16\}\).

### Wall-clock proxy

Each update costs

\[
c_{\mathrm{time}}(q)
=1+0.25\max_{i\in S(q)}\tau_i+0.02q,
\]

with total budget \(W=800\). The coefficients are fixed before execution and
are a simulator proxy rather than measured hardware time.

## Primary metrics

For the primary server-time, fastest-agent setting:

1. optimal \(q^\star\) for every \((\rho,D)\) cell under message budget;
2. risk ratio of the jointly optimal action to the best \(q=32\) action;
3. the same two quantities under the wall-clock proxy;
4. Monte Carlo agreement for the selected optimum and \(q=32\) comparator in
   four pre-specified extreme cells:
   \((\rho,D)\in\{(0,4),(0,16),(0.9,4),(0.9,16)\}\).

The two actions in each Monte Carlo cell are simulated with 10,000
replications and paired only through the deterministic cell definition, not
through shared random numbers.

## Success gates

EXP-005A passes only if all of the following hold:

1. **participation regime change:** under the primary message budget,
   \(q^\star\ge16\) for both independent-noise cells and \(q^\star\le8\) for
   both \(\rho=0.9\) cells;
2. **material high-correlation gain:** in both \(\rho=0.9\) primary message
   cells, optimal risk is at most 0.90 times the best \(q=32\) risk;
3. **wall-clock relevance:** in at least three of the four
   \(\rho\in\{0.6,0.9\}\) cells, wall-clock-optimal \(q^\star\le8\) and risk is
   at most 0.90 times the best \(q=32\) risk;
4. **not only straggler deletion:** with `uniform_rank` selection and primary
   message budget, at least one \(\rho=0.9\) cell has \(q^\star\le8\) and risk
   at most 0.95 times the best \(q=32\) risk;
5. **numerical validity:** every selected action is stable and finite, and each
   Monte Carlo estimate differs from its exact MSE by no more than
   \(\max(5\%,3\text{ standard errors})\).

No gate or coefficient will be changed after looking at the results. A failed
gate remains a failed result and redirects the project away from making
dynamic participation a main accuracy/resource claim.

## Expected outputs

- `results/budget_participation/surface.csv`
- `results/budget_participation/optimal_actions.csv`
- `results/budget_participation/monte_carlo_validation.csv`
- `results/budget_participation/summary.json`
- figures for optimal participation and risk improvement

## Execution

- working directory:
  `experiments/dependence_delay_linear`;
- smoke command:
  `python run_budget_participation.py --output-dir results/smoke/budget_participation --mc-replications 500`;
- primary command:
  `python run_budget_participation.py --output-dir results/budget_participation --mc-replications 10000`;
- hard timeout: 10 minutes;
- environment: local Windows CPU; no GPU required.

## Registered execution result

The smoke run (500 Monte Carlo replications) and primary run (10,000
replications) both exited successfully. The exact surface contained 13,056
actions. Twelve deterministic implementation tests passed.

All five pre-registered gates passed:

1. under the primary message budget, both independent-noise cells selected
   \(q^\star=32\), while both \(\rho=0.9\) cells selected \(q^\star=1\);
2. at \(\rho=0.9\), the optimal-to-all-agent MSE ratios were 0.2598 for
   \(D=4\) and 0.2581 for \(D=16\);
3. all four pre-specified hard wall-clock cells passed the \(q^\star\le8\) and
   ratio-at-most-0.90 condition;
4. both high-correlation `uniform_rank` cells passed, so the primary effect is
   not explained only by dropping the slowest agents;
5. all eight Monte Carlo checks agreed with exact MSE within the registered
   tolerance. The largest relative discrepancy was 2.10%.

The independent same-seed rerun produced byte-identical CSV, JSON, and figure
artifacts for all six retained outputs. Reproducibility is therefore
`REPRODUCIBLE` for the recorded environment.

## Interpretation and next gate

EXP-005A establishes the mechanism required for a resource-aware participation
paper: optimal participation changes sharply when common cross-agent
dependence makes additional messages redundant. The result remains a
model-level oracle surface, not evidence for an implementable adaptive
controller.

The strength of the result also exposes the next falsification target. In every
primary correlated cell, the optimum reached both registered lower boundaries:
\(q=1\) and \(\eta=0.0025\). The one-factor exchangeable model and fixed
resource proxy may therefore make the trade-off unrealistically clean.
EXP-005B must use only selected-agent observations, charge exploration probes,
include clustered and heterogeneous correlation, and compare against the
hindsight oracle under matched resources. No ICML-level controller claim is
made from EXP-005A alone.
