## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: final estimator-class development decision
- Origin Date: 2026-09-08
- Verification Status: PURSUIT EFFICACY BENCHMARK STOPPED
- Version Label: pursuit_raw_factor_final_decision_v1

# Final Pursuit estimator-class decision

## Decision

Pursuit is permanently stopped as the principal efficacy benchmark for Causal
Policy Freshness.  The final permitted estimator change, a parameter-shared
raw-observation convolutional factor critic, failed all four development gates.
No Pursuit calibration, efficacy pilot, formal run, GPU job, or additional
estimator search is authorized.

Pursuit remains useful as an appendix structural stress test: its
state-dependent degree-four support, graph turnover, recipient-local policy
caches, refresh persistence, actor-byte charging, copied-state rollout, and
topology-motion accounting have all been verified.  This decision does not
change those results or the positive finite-state Lyapunov mechanism result.

## Frozen design

- Train seeds: `94500--94515`, 1,024 selected completed packets.
- Validation seeds: `94516--94519`, 256 selected completed packets.
- Test seeds: `94520--94521`, 64 launches, eight independent copied-state
  replicates per candidate.
- Actor path: public local-chase heuristic, drift fraction `0.20`.
- Packet: eight-step launch-score times discounted-return gradient.
- Critic: shared two-layer 12-channel convolutional encoder, 32-dimensional
  embedding, 64-unit self and policy-profile pair heads.
- Early stopping: validation selected-packet return only; best epoch 3 and 39
  epochs executed.
- Gate thresholds were identical to the summary-feature critic and fixed
  before the run.

The ignored artifact is
`tmp/policy_dependency_sync/pursuit_raw_factor_development_v1.json`, SHA-256
`88513722107EB678371EB9225E4A5FD78AF1A40839211F6570D5864A7119CEC1`.
The embedded source hash is
`6bbd53b8a0e50507a162d70bb357a3b3e789485cf4c70812cb7dd2066acbdd44`.
Elapsed local CPU wall time was `1059.17 s`; no HPC4 or GPU was used.

## Results

Validation selected-packet return prediction was no better than a constant
mean (`R^2=-0.009618`).  On untouched replicated conditional means:

| Gate | Threshold | Observed | Result |
|---|---:|---:|---|
| Edge-effect `R^2` | `> 0` | `-0.026293` | fail |
| Nonzero sign accuracy | `>= 0.60` | `0.567568` | fail |
| Best action accuracy, including null | `>= 0.55` | `0.500000` | fail |
| Mean edge effect / mean absolute null alignment | `>= 0.05` | `0.018956` | fail |

The mean absolute edge effect was `0.003461` against mean absolute null
alignment `0.182589`.  Only 34.38% of reference-available launches had a
strictly positive oracle edge.  Prediction error was essentially the same as
predicting zero edge effect, so the critic did not discover a usable refresh
ordering.

## Scientific interpretation

The failure is not evidence that the paired Lyapunov theorem is wrong.  The
theorem says what follows if a predictable lower alignment estimate has a
declared uniform error.  This experiment shows that the standard Pursuit
signal under the declared policy path does not permit the tested selected-
packet critics to make that error nonvacuous while preserving useful actions.

The result also prevents a misleading narrative in which dynamic graph
turnover is presented as empirical learning value merely because it is
structurally present.  Pursuit has topology motion, but its learned edge value
is too weak and poorly identifiable here.

## Replacement-benchmark rule

A replacement may be considered only through a new outcome-free benchmark
contract.  Before reading controller returns, it must establish:

1. a public local policy-dependency factorization with a bounded declared
   degree;
2. a shaped or otherwise statistically informative learning signal for a
   selected-packet centralized critic;
3. exact actor-transition and policy-byte accounting under asynchronous
   launch and receipt;
4. an oracle-value ceiling over a strong online baseline family at matched
   resources;
5. a candidate scan linear in local degree and a dense-game degeneration rule.

Only a benchmark passing these formulation-level gates can receive an efficacy
pilot.  The next work item is benchmark selection and static headroom, not
another Pursuit network.
