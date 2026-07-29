# Experiment 005C: Sparse participation control under regime shifts

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-29
- Verification Status: UNVERIFIED
- Version Label: preregistration_v1

## Objective

This is the final go/no-go experiment for making dynamic agent participation a
main contribution. It tests whether sparse, charged correlation probes can
outperform every fixed participation level when dependence changes within one
training trajectory.

## Dynamic environment

Each run has total message-equivalent budget 32,000 and 16 blocks of budget
2,000. Four consecutive four-block regimes are used:

1. independent: \((\rho_g,\rho_c)=(0,0)\);
2. clustered: \((0,0.6)\);
3. global: \((0.6,0)\);
4. mixed: \((0.3,0.4)\).

The heterogeneous global/cluster loadings, four interleaved clusters, Markov
coefficients, curvature, and deterministic delay profiles are identical to
EXP-005B. Maximum delay is \(D\in\{4,16\}\).

## Sparse controller

At the beginning of every block, `sparse_adaptive` performs four charged probe
updates using the fastest eight agents and fixed probe step size 0.02. Each
probe update costs \(8+4=12\),
so total exploration cost is

\[
16\times4\times12=768,
\]

or 2.4% of total budget.

The controller retains the most recent 12 probe snapshots. From the observed
eight-dimensional residuals it fits three nonnegative scalar second-moment
components: global, within-cluster, and idiosyncratic. Known AR coefficients
convert these components into an estimated long-run variance for
\(q\in\{1,2,4,8,16,32\}\). It then searches the registered 17 scalar step
sizes for the remainder of the current block. It never forms or stores an
\(N\times N\) covariance matrix and never observes a non-participating
gradient outside charged probe updates.

## Policies

- `sparse_adaptive`: sparse moment estimation and blockwise joint
  \((q,\eta)\) selection;
- `piecewise_oracle`: knows the current true dependence components and jointly
  selects \(q,\eta\), without probe cost;
- `fixed_q1_oracle_eta`;
- `fixed_q4_oracle_eta`;
- `fixed_q8_oracle_eta`;
- `fixed_q32_oracle_eta`.

The fixed policies know the true current long-run variance for their registered
\(q\) and choose the best step size each block. They are intentionally stronger
than implementable fixed baselines. The hindsight-best fixed policy is chosen
only after aggregating its pre-specified normalized score.

## Experimental design

- 64 paired seeds, base seed 20260729;
- both delay settings for every seed and policy;
- 161 common budget checkpoints;
- each regime score is its mean squared error over the latter half of that
  regime;
- for every seed, delay, regime, and policy, divide this MSE by the corresponding
  `piecewise_oracle` MSE;
- primary dynamic score: arithmetic mean of these normalized regime scores;
- paired bootstrap resamples seeds and preserves both delays and all regimes;
- 2,000 bootstrap replications.

This within-cell normalization is fixed before execution and prevents
high-variance regimes from dominating solely through measurement scale.

## Success gates

All gates must pass:

1. **best-fixed improvement:** `sparse_adaptive / hindsight-best-fixed`
   normalized dynamic score is at most 0.90 and its paired-bootstrap 95% upper
   endpoint is below 1.0;
2. **oracle proximity:** adaptive normalized dynamic score is at most 1.25 and
   its 95% upper endpoint is below 1.40;
3. **switch response:** by the second block of a regime, selected median
   participation has the correct registered direction in at least three of
   four regimes:
   independent \(q\ge16\), clustered \(q\le8\), global \(q\le4\), and mixed
   \(q\le8\);
4. **exploration budget:** every adaptive trajectory spends exactly 768 probe
   units and no more than 5% of total budget;
5. **accounting and numerical validity:** every trajectory is finite, every
   update is charged, and no policy exceeds its per-block or total budget.

Failure of any gate rejects dynamic participation as the present paper's main
algorithmic contribution. The result will not be repaired by changing regime
weights, normalization, thresholds, probe length, or fixed baselines after
execution.

## Execution

- working directory: `experiments/dependence_delay_linear`;
- smoke:
  `python run_sparse_dynamic.py --output-dir results/smoke/sparse_dynamic --num-seeds 4 --bootstrap-replications 100`;
- primary:
  `python run_sparse_dynamic.py --output-dir results/sparse_dynamic --num-seeds 64 --bootstrap-replications 2000`;
- hard timeout: 15 minutes;
- local Windows CPU; no GPU required.

## Expected outputs

- `per_seed_regime_metrics.csv`;
- `block_actions.csv`;
- `budget_trajectories.csv`;
- `paired_bootstrap_ratios.csv`;
- `summary.json`;
- budget trajectory and block-participation figures.

## Execution outcome

The 4-seed smoke run completed and showed that the code path, accounting, and
artifact generation worked. Its scientific gates failed, but smoke output is
not admissible as primary evidence.

The registered 64-seed primary command exceeded the 15-minute hard timeout.
The command wrapper timed out after approximately 2,228 seconds and left its
Windows Python child process running; that child was identified by its exact
EXP-005C command line and explicitly terminated. No primary output artifact
was produced.

Status:

- primary execution: `TIMEOUT`;
- statistical conclusion: `CANNOT_VERIFY`;
- reproduction: not attempted because no completed primary result exists;
- participation go/no-go decision: unresolved by EXP-005C.

The experiment was not silently retried, reduced to fewer seeds, or granted a
post-hoc longer timeout. A future execution requires an explicitly authorized
new run after performance diagnosis, while preserving the registered
scientific design and recording any computational change as a new execution
version.

## Execution version 2 authorization

On 2026-07-29, the user explicitly authorized performance diagnosis,
semantics-preserving optimization, and a new primary execution. Execution
version 2 freezes every scientific element of preregistration version 1:

- the 64 paired seeds beginning at 20260729 and their random-number streams;
- both delay cells, all six policies, the regime sequence, and all budgets;
- the controller observations, rolling window, candidate actions, and probes;
- all 161 checkpoints, score definitions, bootstrap resampling, and gates.

Permitted computational changes are limited to caching deterministic
quantities, eliminating repeated array construction, vectorizing deterministic
post-processing, and equivalent low-level acceleration. Before a primary run,
version 2 must match version 1 on a paired deterministic equivalence fixture,
including checkpoint errors, block actions, and budget accounting. The primary
command remains unchanged, and the registered 15-minute hard timeout remains
in force.

## Execution version 2 outcome

Execution version 2 completed the registered 64-seed command in 102.5 seconds.
All expected artifacts were produced. A deterministic full rerun completed in
83.6 seconds, and all eight artifacts matched byte-for-byte by SHA-256.

Only the exploration-budget and accounting/numerical-validity gates passed.
The adaptive-to-best-fixed ratio was 2.619 with paired-bootstrap 95% interval
[1.628, 4.080]; the adaptive-to-piecewise-oracle normalized score was 8.226
with interval [6.215, 10.580]; and only one of four switch-response cells
passed. The registered overall decision is therefore **FAIL**.

The validation audit also found that the piecewise oracle itself selected
median \(q=32\) in every regime. Thus EXP-005C rejects the implemented sparse
controller as the present main contribution, but its switch gate does not
support a general conclusion against correlation-adaptive participation.
Detailed statistics and the 11/11 fallacy scan are recorded in
`validation_exp005c.md`.
