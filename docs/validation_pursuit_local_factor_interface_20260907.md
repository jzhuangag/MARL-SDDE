## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: outcome-free formulation validation
- Origin Date: 2026-09-07
- Verification Status: INDEPENDENT STRUCTURAL AUDIT PASSED
- Version Label: pursuit_local_factor_interface_result_v1

# Pursuit local-factor interface qualification

## Result

All five frozen structural gates pass on 32 independent environment seeds and
3,200 joint states. This authorizes derivation of the bounded-factor Lyapunov
algorithm and its performance theorem. It does not authorize a GPU learning
experiment and is not evidence that the algorithm improves return.

| Property | Frozen requirement | Observed | Status |
|---|---:|---:|---|
| Maximum reverse evaluations | at most 6 | 6 | pass |
| Degree-cap truncation | at most 2% | 0.0781% | pass |
| Consecutive graph turnover | at least 30% | 75.6944% | pass |
| Fixed-initial graph edge misses | at least 30% | 56.8622% | pass |
| Nonempty graph states | at least 95% | 99.7188% | pass |

The graph has a mean of 7.8419 directed edges over eight owners. Median
reverse cost is three evaluations per owner action, p90 is four, and the
maximum is six. A typical 100-cycle trajectory visits 66.5 distinct directed
graphs. The mean Jaccard distance between consecutive graphs is 0.2087.

## Provenance and interpretation

- Frozen source commit: `46b235eea5a1de550647c290532076c4a045f621`
- PettingZoo: 1.26.1
- Environment seeds: 91000--91031
- Action seed: 8842
- Raw result SHA-256:
  `c826970648b8d1fe6e8840643af5074407c5c51c3914905685e325299741f2a7`
- A clean second execution produced a byte-identical file with the same hash.
- Pinned upstream source hashes are stored in the machine-readable summary.
- Execution: local CPU; no HPC4/GPU and no `/project` write.

The runner accesses centralized pursuer positions solely to construct the
training-time factor graph. It intentionally discards all rewards returned by
the environment, and it does not instantiate an actor, critic, optimizer, or
learning score. Thus the result establishes state-dependent structural
headroom relative to one fixed graph, not learning-value headroom relative to
a strong online scheduler.

The low-degree guarantee is architectural: at most four visible nearest
neighbors enter an owner factor. The small observed truncation rate shows this
cap rarely removes a naturally visible neighbor on the pinned task, but the
future theorem must still expose factor-approximation error rather than assume
the bounded critic is an exact representation of the unrestricted value.

## Authorized next work

Before any GPU learner, the following must be closed on paper and in CPU unit
tests:

1. a descent bound for delayed owner packets evaluated under a stale local
   policy cache, with explicit Markov and factor-approximation errors;
2. a composite cache/communication Lyapunov drift bound whose minimizer jointly
   determines packet weight and refresh action;
3. queue stability and an average stationarity guarantee against a feasible
   launch-measurable policy;
4. an exact small-game separation showing nonzero equal-resource advantage
   over strong fixed and age-based scheduling.

Only if these four items pass without using deep-benchmark returns may a new
standard Pursuit development learner be frozen.
