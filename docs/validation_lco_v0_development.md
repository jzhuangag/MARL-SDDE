# LCO-V0 Lyapunov-priced dual-control development validation

## Decision

The frozen one-step value-of-information controller **does not survive** its
development gates.  E2, E3, E4, and E6 fail.  This architecture must not be
confirmed, treated as formal evidence, or transferred to a standard MARL/GPU
benchmark.  No threshold, seed, target cell, or comparator was changed after
execution.

The run itself is valid: 1,920 unique paths and 1,966,080 asynchronous events
completed; all rates are finite; the serialized summary is exactly reproduced
from the stored rows; no method exceeds its hard query allowance.

## Frozen outcome

- execution commit: `386ba68`;
- configuration SHA-256:
  `2e9f3fc22961c13a73860a7a937d5d6428ef7683afd515436bce62f625706580`;
- result SHA-256:
  `87f3a9d466689332fab33d38b5ce0757395d2ef2f50eff70f69e9093d1ff2113`;
- rows/cells: 1,920/240;
- development seeds: 83,001--83,008, never eligible for confirmation.

| Frozen metric | Threshold | Noise 0 | Noise 0.05 | Result |
|---|---:|---:|---:|---|
| Mean gain over strong fixed | at least 0.02 | 0.014653 | 0.014579 | fail |
| Median exact gain capture | at least 0.40 | 0.320225 | 0.207192 | fail |
| Improved dynamic-cell fraction | at least 0.75 | 0.59375 | 0.59375 | fail |
| Mean gain over myopic HMM | at least 0.002 | 0.006041 | 0.006064 | pass |
| Information-induced call fraction | at least 0.001 | 0.076771 | 0.074871 | pass |

The low-persistence mean gain is 0.007926 versus its frozen 0.01 gate.  The
low-budget mean gain is positive but only 0.001176.  Both arrival groups are
positive, all selected dynamic cells contract, stationary potential loss is
zero, and budget overshoot is zero.

## What the ablation establishes

The value-of-information term is causally active rather than decorative.  It
creates roughly 7.5% information-induced calls and improves aggregate log rate
over the same hidden-state filter without the information term by about
0.00605.  Nevertheless it spends information calls badly in scarce or
low-rotation regimes:

| Group | Gain over strong fixed | Gain over myopic | Exact-phase headroom |
|---|---:|---:|---:|
| arrival 0.1 | 0.008604 | -0.003917 | 0.034980 |
| rotation fraction 0.25 | 0.000346 | -0.003021 | 0.046591 |
| budget 0.25 | 0.001176 | -0.001134 | 0.033946 |
| persistence 0.80 | 0.007926 | 0.003037 | 0.064615 |
| rotation fraction 0.50 | 0.044817 | 0.032076 | 0.106040 |

Thus the one-step rule is useful when rotations are common, but its local
information value over-explores when queries are scarce or the rotational
state is rare.  Extending the lookahead or multiplying an information bonus
after observing these results would be an unregistered rescue and is not
authorized.

## Next scientific gate

The exact-phase ceiling remains much larger than the realized causal gain, so
the optimization problem has not disappeared.  But two causal sensor
architectures have now failed.  The next step must not be a third heuristic.
It is an outcome-free Bayes-optimal feasibility bound under the *same*
action-dependent observation law and query budget:

1. formulate the hidden two-state geometry problem as a constrained belief
   MDP;
2. compute a numerically certified upper bound on the best attainable
   log-drift improvement using the declared emissions;
3. include the strong fixed policy in the feasible set and charge every
   observation-producing query;
4. stop active sensing if this upper bound lacks the frozen practical
   headroom; only if it is clearly positive may a low-complexity approximation
   be designed.

This separates an information-limited problem from a poor controller without
using more seeds or moving directly to a favorable benchmark.

A clean scientific reproduction was not run because all survival gates were
mandatory and the architecture failed.  The run was local CPU only; no GPU,
HPC4, or remote storage was used.
