# MaMuJoCo fast extension preregistration

## Material Passport

- Experiment: `TSP-MARL-MAMUJOCO-DEV-001`
- Mode: prospective development
- Status: frozen before execution
- Task: MaMuJoCo `HalfCheetah-v2/2x3`
- Scientific role: candidate second nonlinear environment for the TSP manuscript

## Motivation and separation from the stopped multi-task scan

The completed two-task Stage-A scan is permanently recorded as a stopped conjunction because Speaker--Listener did not clear its task-level headroom threshold.
That decision and its artifacts are unchanged.
The same preregistered scan separately found `3.793433%` endpoint-oracle headroom and an independent/shared participation reversal on HalfCheetah.
Those values are treated only as development information motivating this new task-specific experiment; they are not reused as confirmation evidence.

This experiment asks a different and narrower question: can the already specified, return-free, fully charged Lyapunov controller realize the observed participation value on a continuous-control task when compared with the complete fixed catalogue?

## Frozen design

- Algorithm: MAPPO from HARL commit `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`.
- Task: `HalfCheetah-v2`, partition `2x3`, shared policy parameters, continuous actions.
- Dependence regimes: independent and shared rollout innovations, each preserving the per-worker Gaussian action marginal.
- Fixed baselines: `q` in `{1,2,4,8}`.
- Controller candidates: the same complete catalogue.
- Physical budgets: 5,000,000 messages and 1,000,000 environment ticks.
- Probe: 16 fully charged, non-learning blocks with `q_probe=8` and block length 200.
- Controller score: the certified dependence variance factor divided by the exact number of post-probe updates feasible under both budgets.
- Development seeds: training `125001--125004`; disjoint probe seeds `126001--126004`.
- Total runs: 40.

The probe uses 38,400 messages and 3,200 environment ticks, or `0.768%` and `0.32%` of the respective budgets.
No evaluation return enters the participation decision.

## Mandatory gates

1. All 40 cells are unique, finite, exactly budgeted, and use the clean pinned upstream commit.
2. The complete fixed catalogue is present in both regimes.
3. In each regime, controller mean return AUC is within two percent of the best fixed-`q` mean return AUC.
4. Across the equally weighted regimes, controller mean return AUC improves by at least one percent over the strongest single fixed `q` selected across both regimes.
5. Probe-message fraction is at most one percent and scalar selection overhead is at most two percent of training time.
6. Two independent analyzer executions are byte-identical.

Selection of a particular `q` is diagnostic rather than a mandatory gate.
If every gate passes, a separate confirmation protocol will freeze new seeds and a paired one-sided confidence test before confirmation runs begin.
Any failed gate stops this extension without changing the task, controller, seeds, or thresholds.

