# Experiment 001: Dependence–delay go/no-go test

## Material Passport

- Artifact type: code experiment plan and execution record
- Experiment ID: `EXP-001-dependence-delay-linear`
- Project: SDDE multi-agent Markov learning
- Status: implementation ready; results pending
- Local scope:
  `experiments/dependence_delay_linear/`
- Evidence policy: exact linear-system calculations are primary; Monte Carlo is
  an independent numerical cross-check

## Research question

When agent updates share a persistent Markov factor, does the useful
parallelism saturate? When agents also have heterogeneous staleness, is there a
nontrivial optimal number of accepted agents?

## Hypotheses

- H1: Under independent agent noise, increasing the number of agents markedly
  reduces finite-horizon mean-square error.
- H2: A common Markov factor creates a non-vanishing variance component, so the
  marginal benefit of additional agents saturates.
- H3: When additional agents are systematically staler, the best finite-horizon
  risk can occur at an interior participation level.
- H4: A participation and step-size rule designed under cross-agent
  independence is detectably suboptimal in the correlated regime.

## Design

The experiment compares two deterministic delay profiles:

1. synchronous agents, isolating correlation-limited speedup;
2. fastest-first heterogeneous agents, testing the participation–staleness
   tradeoff.

For every setting, a grid search over the scalar step size computes:

- augmented-system spectral radius;
- finite-horizon squared bias and variance;
- finite-horizon MSE;
- stationary MSE.

The finite-horizon covariance is computed from the discrete Lyapunov solution,
not estimated through repeated trajectories. Three selected settings are then
simulated independently to detect implementation or algebra errors.

## Decision criteria

- Correlation saturation: high-correlation \(q=1\)-to-\(q=32\) improvement is
  less than half the independent-noise improvement.
- Interior optimum: at \(\rho=0.9\), the joint oracle uses fewer than 32 agents
  under heterogeneous staleness.
- Controller relevance: the delay-only choice has at least 1.2 times the joint
  oracle's MSE at \(\rho=0.9\).
- Reproducibility: all Monte Carlo checks have at most 5% relative error from
  the exact value.

## Interpretation boundary

Passing establishes that the proposed mechanism is present in a controlled
linear model. It does not establish novelty, minimax optimality, or performance
for policy-dependent nonlinear MARL. Failing a criterion is useful evidence
that the model, controller target, or paper claim should be revised before
larger experiments.

## Execution record

Baseline completed on 2026-07-29 with the registered `sample_time` alignment.
The machine-readable record is in
`experiments/dependence_delay_linear/results/baseline/summary.json`.

Observed gate outcomes:

- correlation saturation: PASS. The \(q=1\)-to-\(q=32\) finite-horizon gain was
  \(22.60\times\) under independent noise but only \(1.0034\times\) at
  \(\rho=0.9\);
- interior parallelism optimum: FAIL. The joint oracle selected \(q=32\);
- delay-only suboptimality: PASS. At \(\rho=0.9\), the dependence-blind choice
  had \(1.8745\times\) the joint oracle's MSE;
- exact versus Monte Carlo agreement: PASS. The maximum relative discrepancy
  over the three validation points was \(2.42\%\), and every exact value was
  inside its corresponding Monte Carlo 95% interval.

The high-correlation joint oracle still selected all 32 agents. Inspection
showed that the delay-only gap was produced primarily by step-size mismatch,
not by a different participation decision. Consequently, EXP-001 supports a
dependence-aware step-size mechanism but does not yet support the stronger
claim that more agents can be harmful.

## Post-baseline sensitivity

EXP-002 changes only the temporal alignment of the common factor. In EXP-001,
the common factor is evaluated at each worker's delayed sample time; averaging
different delays therefore also averages different points of the common
Markov trajectory. EXP-002 instead evaluates the common factor at the shared
server/environment time while retaining stale parameter copies. This
distinguishes time-staggering decorrelation from genuinely shared
contemporaneous environmental noise. EXP-002 is exploratory and cannot replace
the failed EXP-001 interior-optimum gate.

EXP-002 again selected all 32 agents at the 500-step jointly tuned operating
point. Its dependence-blind risk ratio was \(1.8881\), so changing the common
factor alignment did not recover an interior jointly tuned optimum.
