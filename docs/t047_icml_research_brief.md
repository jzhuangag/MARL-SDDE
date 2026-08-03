# T-047 ICML 2027 research brief

## Research question

Under fixed message and environment budgets, can a low-complexity,
probe-separated learner adapt a time-varying participation schedule to
cross-agent correlation and the finite-resource learning phase, while
retaining a rigorous Markov-noise guarantee and improving
Polyak--Ruppert-averaged temporal-difference prediction over the strongest
deployable fixed-participation baseline on prospectively frozen standard
reinforcement-learning tasks?

## Publication thesis

Parallel Markov learning has two coupled sources of diminishing returns.
Cross-agent dependence reduces the information supplied by a larger batch,
whereas communication and delay reduce the number and age of usable updates.
A single fixed participation level cannot generally balance transient
contraction and late-stage variance across correlation regimes. The proposed
method therefore uses independent, fully charged probes to select a
resource-feasible schedule from a frozen library containing both fixed and
two-stage participation policies. The central contribution is not adaptive
batch size alone; it is a finite-budget correlation--delay law, an explicit
adaptation cost, and a controller whose online decision is a constant-size
affine-risk table scan.

## Planned theorem package

1. **Exact scheduled phase law.** For delayed additive vector linear
   stochastic approximation, prove the finite-horizon risk identity for any
   deterministic prefix-participation schedule. The full cross-time
   covariance depends on the overlap of the participating agent sets.
2. **Affine correlation reduction.** Show that every frozen schedule has
   exact risk \(R_\pi(\rho)=a_\pi+b_\pi\rho\), enabling an \(O(|\Pi|)\)
   selector after offline coefficient computation.
3. **Probe-cost oracle inequality.** Under certified mixing and a separated
   schedule gap, bound excess risk by the probe opportunity cost, correlation
   estimation radius, and selection error. Match the dependence on
   correlation, mixing, delay, and both budgets to the T-038 lower bound.
4. **Conditional affine-TD transfer.** Apply the T-042 decomposition and T-046
   small-gain envelope to state a nonvacuous finite-state temporal-difference
   corollary. The additive theorem remains the main unrestricted result.

## Algorithmic contract

The controller uses a maximum agent catalogue \(Q\), stride catalogue \(B\),
and a finite schedule library \(\Pi\). It must satisfy all of the following.

- Probe and learning streams are sample-split in the theorem-facing track.
- Probe messages and environment interactions are charged to the same total
  budgets as every baseline.
- The correlation statistic is computed from scalar gradient projections or
  pairwise gradient inner products, not exact state collisions or source IDs.
- Unknown mixing is not claimed. A public certificate or an independent upper
  confidence certificate bounded away from one is required.
- The post-probe selector uses only confidence bounds, remaining resources,
  delay summaries, and the frozen affine-risk table. Held-out prediction
  error, true correlation, source labels, and another policy's outcomes are
  forbidden inputs.
- Online arithmetic is \(O(qd+|\Pi|)\), with no Hessian inverse,
  covariance-matrix inverse, or preconditioner.
- A fallback statement includes the already-paid probe opportunity cost; it
  is not described as strict no-harm relative to a no-probe oracle.

## Comparison contract

The primary baseline is the strongest deployable fixed \((q,b)\) selected on
disjoint pilot seeds within each task--budget pair and then frozen before
formal evaluation. It may not condition on formal-test correlation, delay, or
outcomes. A cellwise fixed-action oracle is reported only as an unattainable
diagnostic ceiling. The empirical controller may claim improvement over the
deployable baseline and oracle-normalized adaptation cost; it may not claim
uniform dominance over the cellwise oracle.

## Prospective evidence architecture

### EXP-021A: exact and sampled CPU temporal-difference benchmark

The candidate task catalogue is fixed before learning outcomes:

- `FrozenLake-v1`, 8-by-8 slippery map;
- `CliffWalking-v0`;
- `Taxi-v3`.

Every task uses a deterministic, publicly generated epsilon-soft evaluation
policy, regenerative terminal handling, and unchanged-law common/private
random streams. Exact-kernel calculations use tabular features when
computationally feasible; Taxi remains in the sampled population even if its
full exact matrix calculation is omitted for cost. Resource rays are functions
of public mixing and drift certificates, not pilot errors. No task may be
removed after a learning outcome is observed.

### EXP-021B: nonlinear standard-benchmark breadth

Only an EXP-021A pass can authorize EXP-021B. Its primary suite is all five
MinAtar games with the upstream full action set. Common/private coupling acts
on complete sticky-action, spawn, reward, and reset random streams, preserving
each actor's marginal environment law. A later Procgen transfer is optional
and requires its own preregistration. The official MinAtar testbed was designed
for lower-cost, reproducible behavioral RL experiments, while Procgen provides
sixteen procedurally generated tasks for sample-efficiency and generalization
evaluation:

- https://arxiv.org/abs/1903.03176
- https://github.com/kenjyoung/MinAtar
- https://proceedings.mlr.press/v119/cobbe20a.html

## Mandatory progression gates

EXP-021A pilot is authorized only after a separate preregistration commit fixes
the runner, analyzer, task hashes, schedule library, budgets, seeds, metrics,
and gates. Formal execution requires every pilot gate below.

1. All resource accounting, marginal-law, mixing, and leakage tests pass.
2. The exact scheduled-risk implementation matches constant-participation
   T-037 and independent Monte Carlo calculations.
3. A full-cost dynamic-schedule oracle improves the strongest deployable fixed
   baseline by at least 5% in aggregate and strictly in at least 50% of the
   registered message-binding cells.
4. The learned controller achieves at least 3% aggregate pilot improvement,
   has a controller/baseline risk ratio at most 1.02 on the registered
   no-value population, and selects at least two distinct schedules.
5. Correlation-response and delay-response directions each hold in at least
   80% of their registered comparisons.
6. Controller wall time is at most 10% of total measured learning time.
7. A clean rerun is byte-identical for deterministic artifacts and satisfies
   the frozen stochastic reproducibility tolerance for sampled metrics.

Any mandatory failure stops formal seeds without changing tasks, thresholds,
resource rays, or analysis populations. Failure of EXP-021A stops EXP-021B.

## Novelty boundary

The paper must distinguish its contribution from independent-agent federated
reinforcement-learning speedup, delayed Markov stochastic approximation,
controlled sensing, and adaptive batch scaling. In particular, the 2026
Adaptive Batch Scaling preprint changes rollout length using policy
nonstationarity on the Arcade Learning Environment. The present claim must
remain the interaction of cross-agent correlation, time-varying participation,
delay, dual budgets, and an explicit identification cost; policy-stability
batch adaptation is a comparator rather than novelty evidence.

Primary comparison records:

- https://proceedings.mlr.press/v162/khodadadian22a.html
- https://proceedings.mlr.press/v238/adibi24a.html
- https://arxiv.org/abs/2605.21557

## Stage-1 decision

The research question is technically coherent and differs from the stopped
EXP-017A/T-045A line in both theorem object and algorithm. It is authorized to
proceed to T-048 proof closure and an outcome-free EXP-021A static design. No
sampled pilot, formal seed, GPU job, or HPC4 job is authorized by this brief.
