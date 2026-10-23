# T-031 ICML 2027 reframe

## Decision

The homogeneous scalar-participation line is stopped.  The new candidate main
line is

> **Agents Are Not Samples: Fresh-Diversity-Limited Speedups for Delayed
> Markov Learning**

The central object is no longer the number of participating agents by itself.
It is the effective information in a predictable subset of jointly dependent
Markov streams, after accounting for staleness and resource cost.

The decisive comparison first fixes the per-round participation count `m` and
chooses which `m` agents to use from a larger pool.  Equal-size subsets have
the same update horizon and first-order resource cost, so a diversity gain is
not cancelled by fewer contraction steps as it was in EXP-019A.  The outer
algorithm may then enumerate a small public grid of `m` values.

For a subset `S`, let `Omega_S` denote the long-run covariance of the agents'
stationary innovations and define

\[
 v(S,a)=a^\top\Omega_S a,
 \qquad
 n_{\rm eff}(S,a)=\frac{\sigma_{\rm ref}^2}{v(S,a)}.
\]

Under equal weights, independent homogeneous streams recover
`n_eff=q`; equicorrelated streams recover
`q/[1+(q-1)rho]`.  With a dependency graph, two subsets with the same `q`
can have different effective parallelism.  This is the sense in which agent
count remains in the paper without being the sole decision variable.

## Research question

Can a low-complexity graph-aware scheduler attain, up to constants, the
minimax finite-time error determined jointly by long-run cross-agent
dependence and staleness in delayed multi-agent Markov stochastic
approximation?

### FINER assessment

| Criterion | Score | Reason |
|---|---:|---|
| Feasible | 4/5 | Existing affine delayed-SA, covariance-calibration, and adaptive change-of-measure components are reusable. |
| Interesting | 5/5 | It asks when more agents are genuinely more data, rather than assuming linear speedup. |
| Novel | 4/5 | The defensible gap is cross-agent Markov dependence plus delay and matching rates, not generic client selection. |
| Ethical | 5/5 | No human subjects or sensitive data are required. |
| Relevant | 5/5 | The question applies to parallel RL, distributed SA, and correlated simulators/sensors. |
| **Average** | **4.6/5** | Proceed, subject to the stop gates below. |

## Scope

In scope:

- centralized cooperative learning of one common fixed point or fixed policy;
- jointly stationary Markov streams with a sparse or block dependence
  certificate;
- predictable subset selection, heterogeneous delays, and message,
  environment, and wall-clock accounting;
- scalar-step delayed affine SA/TD as the main theorem class;
- a separately delimited nonlinear transfer experiment.

Out of scope:

- strategic Markov games, equilibria, or agent-specific objectives;
- unrestricted unknown mixing or an unrestricted dense covariance graph;
- Hessian inverses, covariance inverses, dense preconditioners, or hidden
  semidefinite programs;
- actor--critic convergence claims unless separately proved;
- reviving EXP-019A or tuning its failed scalar-q selector.

## Why this is not generic client selection

Client selection based on data heterogeneity, loss correlation, gradient
representativeness, clustering, or learned rankings is already crowded by
[Clustered Sampling](https://proceedings.mlr.press/v139/fraboni21a.html),
[DELTA](https://proceedings.neurips.cc/paper_files/paper/2023/hash/949c57d30f8791e3ae42646081b3c102-Abstract-Conference.html),
[Fed-CBS](https://proceedings.mlr.press/v202/zhang23y.html),
[HiCS-FL](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7886b9bafe76c52fd568db10ff9772df-Abstract-Conference.html),
and [FedRank](https://proceedings.mlr.press/v235/tian24d.html).  T-031 instead
keeps every agent's target operator fixed and studies stochastic dependence
among their Markov innovations.  Its necessary novelty is a tight
dependence-adjusted speedup law, a count-only lower bound, and a selector whose
guarantee includes nonseparable dependence--delay interaction.

This gap is visible relative to
[Federated RL under Markovian Sampling](https://proceedings.mlr.press/v162/khodadadian22a.html),
which proves linear speedup under cross-agent independence, and
[Delayed Markov SA](https://proceedings.mlr.press/v238/adibi24a.html), which
adapts to delay without solving identity-aware selection under dependent
streams.  Heterogeneous state-action coverage in
[federated Q-learning](https://proceedings.mlr.press/v202/woo23a.html) is also
different from redundant stochastic innovations with a common target.

## Competing routes rejected as main lines

1. A transient small-q/large-q schedule is an evaluation or ablation only.
   Adaptive batch-size and critical-batch-size questions are already mature,
   and the current project has no nonvacuous theorem score for this route.
2. Active correlation shaping or reset scheduling is an optional extension.
   It overlaps with
   [adaptive antithetic sampling](https://proceedings.mlr.press/v97/ren19b.html)
   and
   [staggered resets](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c13ba9a39b50484d65f969475ee267ae-Abstract-Conference.html).
3. The old variance-only scalar-q controller is permanently excluded as a
   positive algorithmic contribution.

## ICML acceptance conditions

All of the following are mandatory:

1. a finite-time upper bound with explicit long-run cross-agent covariance,
   mixing, identity-specific delay, subset, and budget dependence;
2. a matching lower bound on a Gaussian Markov subclass;
3. a count-only impossibility result for equal-size subsets;
4. an exact block-class separable-convex allocation theorem, followed only
   then by a constant-factor sparse-graph extension, with no dense inverse;
5. a predictability and selection-bias argument for any learned graph; the
   positive online result may be restricted to certified separated classes;
6. an SDDE or delay-SDE result only if a rigorous discrete/continuous bridge
   is proved and its Lyapunov--Krasovskii generator yields the scheduler score;
7. prospective actual-learning gain of at least 10% against the strongest
   task-by-budget fixed-subset baseline in aggregate, with strict improvement
   in at least 70% of preregistered active cells;
8. at least one naturally correlated benchmark, not only injected common
   factors, and a nonlinear transfer on two suites before submission;
9. controller overhead below 10%, complete resource accounting, and new
   formal seeds.

## Stop conditions

The ICML line stops if any of these holds:

- the exact full-risk CPU ceiling is below 15%;
- actual-learning gain is below 5% even when the full-risk ceiling passes;
- only a synthetic common-factor benchmark benefits;
- the lower bound or selector approximation theorem cannot be closed;
- the algorithm reduces to DELTA/FedCor plus an additive delay penalty;
- a dense inverse or retrospective task/gate modification is needed;
- the SDDE layer has no proved bridge to the discrete algorithm.

## Evidence inherited from the previous line

Reusable evidence is limited to the delayed affine proof machinery,
EXP-018B fixed-parameter covariance calibration, AC-7 adaptive
change-of-measure, the unknown-mixing impossibility, and budget-accounting
infrastructure.  EXP-019A and EXP-017A remain negative boundary evidence and
must not be presented as support for the new algorithm.
