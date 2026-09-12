# TSP-MARL-SMACV2-DEV-001 preregistration

## Question and status

This is a development-only external validation of the frozen learning-value
mechanism on a materially larger CTDE problem.  It asks whether a fully
charged, return-free dependence probe can choose between serial and parallel
MAPPO data collection on SMACv2 `terran_10_vs_10`, and thereby outperform the
stronger single fixed participation level under a balanced mixture of
independent and maximally coupled rollout streams.

No SMACv2 trajectory, return, win rate, or learned parameter was observed when
this plan was written.  The preceding `simple_reference_v2` development result
is retained unchanged and is not used to lower any threshold.  This experiment
does not alter the affine theorem, TSP-MARL-CONF-001, or the manuscript unless
all development gates pass and a later confirmation is separately
preregistered.

SMACv2 is chosen for scientific rather than numerical reasons.  Its official
NeurIPS benchmark paper identifies procedural generation, evaluation on unseen
settings from the same distribution, meaningful stochasticity, and partial
observability as design goals.  These properties make dependence among
parallel Markov rollouts consequential in a way that a rapidly saturating
small MPE task need not.  HARL provides the pinned MAPPO CTDE implementation
used here.

Primary sources:

- SMACv2 paper: <https://proceedings.neurips.cc/paper_files/paper/2023/file/764c18ad230f9e7bf6a77ffc2312c55e-Paper-Datasets_and_Benchmarks.pdf>
- SMACv2 repository: <https://github.com/oxwhirl/smacv2>
- HARL repository: <https://github.com/PKU-MARL/HARL>

## Frozen environment and learning configuration

- Environment: SMACv2 `terran_10_vs_10`, ten decentralized execution agents
  against ten enemies, 200-step episode limit.
- Learner: parameter-shared MAPPO actor with a centralized critic; execution
  remains decentralized.
- Rollout participation is a systems variable: `q` is the number of parallel
  environment workers, not the number of strategic agents in the game.
- Candidate actions: `q in {1,8}`.
- Network: two 128-unit ReLU layers; actor and critic learning rates `5e-4`;
  five actor and critic epochs; ValueNorm enabled.
- The task distribution, learner, optimizer, and evaluation law are identical
  across methods and dependence regimes.

The two dependence regimes change only the coupling across rollout workers.
For `independent`, cyclic worker-seed permutations and ordinary categorical
sampling preserve each worker's seed multiset.  For `shared`, workers receive
the same environment seed and the same inverse-CDF categorical uniform.  Each
worker therefore retains the same marginal law while cross-worker dependence
changes.  Evaluation always uses independent deterministic HARL streams.

## Frozen controller and resource geometry

The controller collects 64 separated, non-updating probe blocks at `q=8`,
computes the simultaneous upper confidence bound on average pairwise residual
correlation, and minimizes

```text
(h/L + q) * (rho_upper + (1-rho_upper)/q)
```

over `q in {1,8}`.  With `L=200` and `h=800`, the decision boundary remains
exactly `rho_upper=1/3`; this preserves the dimensionless Lyapunov tradeoff of
the confirmed MPE experiment rather than transplanting its raw costs.

Both probe and learning consume the frozen budgets:

- message budget: 16,000,000 units;
- environment-time budget: 3,000,000 ticks;
- block cost: `800+200q` message units and 200 environment ticks;
- probe cost: 153,600 message units and 12,800 environment ticks;
- fixed `q=1`: 15,000 updates and 3,000,000 actor transitions;
- fixed `q=8`: 6,666 updates and 10,665,600 actor transitions.

Thus the comparison keeps the value of parallel collection and its
communication cost visible without using wall-clock time as the primary
metric.

## Outcomes and analysis

There are four new training seeds and four disjoint probe seeds.  Each
dependence regime contains the controller, fixed `q=1`, and fixed `q=8`, for 24
runs total.  Evaluation uses 64 deterministic episodes on eight evaluation
threads every 250 learner updates.

The primary outcome is the area under evaluation win rate against the maximum
of cumulative message-budget and environment-budget fractions, interpolated
on a frozen 21-point grid without smoothing.  The static comparator is selected
from `q in {1,8}` using the equal-regime mean primary AUC, with exact ties
broken toward `q=1`.  Secondary outcomes are return AUC, terminal win rate,
selection, resource cost, and arithmetic overhead.

All mandatory gates are defined in
`experiments/marl_smacv2_development.json`.  The scientific gates require:

1. at least three of four `q=8` decisions under independent coupling and at
   least three of four `q=1` decisions under shared coupling;
2. at least 0.05 absolute win-rate-AUC headroom for the regime-wise fixed
   oracle over the strong static action;
3. at least 0.025 absolute equal-mixture win-rate-AUC gain for the controller
   over the strong static action;
4. no worse than a 0.02 absolute per-regime primary-AUC gap from the matching
   fixed action;
5. consistent shared-regime direction for return AUC; and
6. exact budgets, marginal-law audit, at most 1.1% probe messages, at most 5%
   scalar selection overhead, and byte-identical analysis replay.

Any failed gate stops this line without replacement seeds, altered thresholds,
or a confirmatory claim.  Passing permits only a new confirmation
preregistration with new seeds and an intermediate-dependence condition.

## Execution and storage

The code and outputs are isolated under
`/scratch/jzhuangag/MARL-SDDE-TSP-SMACV2-DEV-001`.  No experiment output may be
written to `/home` or `/project`.  A two-cell integration smoke may precede the
24 scientific runs, but it must use separate seeds and paths and is excluded
from every statistic.  Source commit, environment package versions, runner and
analyzer SHA-256 values, Slurm commands, job IDs, logs, and result hashes must
be recorded before validation.
