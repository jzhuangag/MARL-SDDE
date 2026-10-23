# TSP-MARL-DEV-002 outcome-free development plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-10
- Verification Status: PREREGISTERED-DESIGN
- Version Label: development_v1

## Research question

Can an observable, low-complexity Lyapunov rule preserve the strong fixed
rollout participation level under weak cross-worker dependence while switching
to a smaller participation level when common randomness destroys effective
sample size, after charging every probe and learning trajectory to the same
communication and environment budgets?

This is the narrow external policy-learning test needed by the TSP manuscript.
It does not assert global convergence of nonlinear MAPPO.  It tests whether the
paper's variance--resource mechanism transfers to a standard CTDE benchmark in
the direction predicted by the Lyapunov model.

## Relation to the immutable predecessor

`TSP-MARL-DEV-001` remains a stopped, one-seed fixed-action scan.  Its registered
gate required strict oracle improvement in both coupling regimes and failed
because `(q, eta)=(8, 0.0005)` was simultaneously the global strong action and
the independent-regime oracle.  The shared-regime oracle was
`(q, eta)=(1, 0.0005)`, with an 8.634% descriptive AUC gain, and the aggregate
oracle headroom was 5.033%.

DEV-002 does not change or reinterpret that gate.  It asks the distinct Pareto
question implied by D0: retain the strong action in the regime where it is
already optimal, and improve the regime where dependence changes the optimum.
All DEV-002 code, seeds, budgets, actions, metrics, and stopping gates are
frozen before observing any DEV-002 return.

## Observable controller

Each probe block launches `q_probe=8` complete joint-policy rollouts for
`L=25` environment ticks.  The actor and critic are not updated.  For rollout
worker `j` and block `s`, the controller records

```
f[s,j] = mean_{time, strategic agent} (return - unnormalized critic prediction).
```

The 128-by-8 fingerprint matrix is standardized columnwise.  Its average
pairwise cross-product gives `rho_hat`; a one-sided normal upper proxy
`rho_upper` is used conservatively for the development decision.  This proxy is
not presented as a time-uniform Markov confidence sequence or as a theorem for
nonlinear MAPPO.  A confirmatory experiment is forbidden unless a later audit
either proves the required coverage under explicit assumptions or labels the
mechanism as an empirical plug-in rule.

Under an equicorrelated stochastic-gradient model, the variance of a `q`-worker
average is proportional to

```
v(q,rho) = rho + (1-rho)/q.
```

One learning block costs `h+qL` messages.  The quadratic Lyapunov drift's
message-limited stochastic term is therefore proportional to

```
J(q;rho_upper) = (h/L + q) * v(q,rho_upper).
```

The controller minimizes `J` over the registered set `{1,8}`, breaking ties
toward smaller `q`.  With `h/L=4`, the exact switch boundary between `q=1` and
`q=8` is `rho_upper=1/3`.  The decision uses no reward return, evaluation
metric, or D0 outcome lookup.  It is `O(Sq^2)` once for `S=128,q=8`, followed by
constant-time action selection; the learning-time overhead is expected to be
negligible.

The critic and actor learning rates are fixed at `0.0005`: both D0 regime
oracles selected this value, so DEV-002 tests the dependence mechanism without
adding a second adaptive degree of freedom.

## Full resource charging

For probe blocks `S`, probe participation `q_p`, and selected learning
participation `q`,

```
probe_messages = S (h + q_p L),
probe_environment = S L,
N_train(q) = min(
  floor((B_msg - probe_messages)/(h+qL)),
  floor((B_env - probe_environment)/L)
).
```

Both probe and learning costs are included on the horizontal axis.  Evaluation
episodes are identical across methods, logged separately, and excluded from
the training budgets.  No probe gradient is reused for learning.

## Frozen experiment

- Upstream: HARL commit `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`.
- Algorithm/task: MAPPO on discrete PettingZoo MPE `simple_spread_v2`.
- Training: CTDE with parameter-shared decentralized actors and a centralized
  critic; deterministic decentralized evaluation.
- Coupling: marginal-preserving `independent` and `shared` rollout streams.
- Methods: Lyapunov probe-then-commit, fixed `q=1`, fixed `q=8`.
- Budgets: `B_msg=5,000,000`, `B_env=1,000,000`, `h=100`, `L=25`.
- Probe: `q_p=8`, 128 blocks, no parameter update.
- Training seeds: `94001, 94002, 94003, 94004`.
- Disjoint probe seeds: `95001, 95002, 95003, 95004`.
- Runs: 2 regimes x 4 seeds x 3 methods = 24.
- Primary metric: deterministic team-return AUC on 21 equally spaced points of
  the exactly charged maximum budget fraction.
- Inference: development point estimates only; no p-value, confidence claim,
  or manuscript figure.

The machine-readable source of truth is
`TSP/experiments/marl_probe_commit_development.json`, SHA-256
`3082cd544596bf13592168034ea7ff2e936027d12d9b9592e03f3c7298ae9fbd`.

## Mandatory stopping gates

All gates are conjunctive:

1. exactly 24 unique finite cells; exact dual-budget accounting; clean pinned
   upstream in every cell;
2. select `q=8` in at least 3/4 independent runs;
3. select `q=1` in at least 3/4 shared runs;
4. independent controller AUC relative to fixed `q=8` at least -2%;
5. shared controller AUC relative to fixed `q=8` at least +5%;
6. equal-weight mixture AUC relative to fixed `q=8` at least +2.5%;
7. probe messages at most 1% of the message budget;
8. scalar selection time at most 5% of learning time;
9. the existing marginal-preserving coupling audit passes.

Any failure stops this extension.  It cannot be rescued by changing seeds,
thresholds, task, actions, probe length, or analysis.  A pass authorizes only a
separate confirmation preregistration with new seeds and a theory-coverage
decision; it is not itself formal evidence.

## Reproducibility and integrity

- Controller SHA-256:
  `14e1f3941f734a7652726251aacb7f59fb21485b9e1387cf29100722d0c2b9c9`.
- Runner SHA-256:
  `732b859408d4116729b81323280f842a5676f79de4ed69111d15d4d8b8be0877`.
- Analyzer SHA-256:
  `36b60621c680b0a6eab2798d8c434a93bb3c062664baf91a1d9b8648fcec4ad4`.
- Targeted pre-outcome tests: 13 passed.
- The runner refuses to overwrite a controller output root.
- The analyzer stops on missing or duplicate cells instead of averaging them.
- Raw trajectory fingerprints, progress, metadata, commands, logs, and SHA-256
  manifests must remain under the isolated scratch root.

## Compute plan

The workload is 24 A30 allocations.  Based on DEV-001, each fixed run should
take roughly 20--60 minutes; controller runs add 3,200 fully charged probe
environment ticks.  Execution belongs under
`/scratch/jzhuangag/MARL-SDDE-TSP-MARL-DEV-002`; no scientific output is written
to `/project` or `/home`.  A non-scientific interface smoke precedes the array.
The array is staged: run controller indices 0--7 first.  If their registered
selection, accounting, coupling, or overhead gates already fail, stop without
spending the fixed-baseline allocations.  Only after those observable gates
pass may indices 8--23 complete the return comparison.
