# Packet-debt exact-game headroom development scan

Date: 2026-09-06

Status: outcome-free development evidence.  The scan was not preregistered,
does not update a policy, and cannot serve as formal paper evidence.

Subsequent actual-learning development stopped this online-horizon/unsigned-
debt branch despite the positive certificate result; see
`signed_policy_graph_development_20260906.md`.

## Question

Does context-adaptive joint selection of rollout horizon and policy-version
graph have enough theorem-facing value to justify building the full
random-delay controller, even against a strong resource-matched online
baseline?

## Model and comparator

The exact cooperative quadratic model has six policy blocks.  For each owner,
a lazy directed Markov chain moves its active donor around the other five
blocks.  A finite rollout averages the local interaction rows.  The slow/noisy
mode has move probability 0.1, innovation variance 1.0, and temporal
correlation 0.7.  The fast/clean mode has values 0.8, 0.05, and 0.1.  The
candidate horizons are 2, 4, and 8.  Every candidate is charged for two
independent trajectories: one receipt-control trajectory and one update
trajectory.  Environment cost is therefore `2H`, not `H`.

This construction makes both sides of the horizon decision endogenous:
averaging reduces Markov/noise error, while a longer horizon expands the
prospective policy-dependency cone.  Three fixed cache-lag profiles, every
owner, and every launch offset produce 180 contexts.

For each of 36 predeclared message/environment price pairs, the dynamic rule
exactly minimizes packet debt plus resource price.  Its comparator is stronger
than a static graph: for each fixed horizon, the graph may still adapt to the
full launch context using the same exact rule.  The final baseline is the
optimal convex mixture over all three fixed-horizon adaptive-graph policies
and their no-refresh variants, constrained not to exceed the dynamic rule's
own mean actor-transition and message costs.  The mixture is solved by linear
programming.

## Development result

| Metric | Observation |
|---|---:|
| contexts | 180 |
| resource-price cells | 36 |
| median dynamic debt gain | 25.1276% |
| maximum dynamic debt gain | 33.6634% |
| cells with at least 5% gain | 35/36 = 97.22% |
| cells with at least 10% gain | 33/36 = 91.67% |
| minimum gain | 4.6909% |
| slow/noisy mean horizon above fast/clean | 35/36 cells |
| median slow-minus-fast mean horizon | 2.1778 cycles |
| distinct context graph supports per cell | 32--100 |
| median dynamic actor-transition cost | 13.0222 |
| median dynamic message cost | 1.4361 |

The gain is not produced by giving the comparator fewer resources.  For every
cell, the convex fixed envelope is optimized under the dynamic policy's two
observed mean costs.  It is also not merely a graph-vs-no-graph comparison,
because fixed-horizon comparator graphs are themselves context adaptive.

## Interpretation and limitation

This scan passed the internal design target of approximately 10% median
theorem-facing headroom and therefore justified the completed actual-learning
development check.
That next check failed: context-dependent certificate debt did not predict
signed accumulated learning progress.
The result is retained as a certificate-phase diagnostic, not as support for
the final controller.

The two regimes are deliberately constructed to expose the predicted phase,
so this is an existence/feasibility result, not evidence that ordinary MARL
benchmarks share the same distribution.  It measures certified packet debt,
not return.  The surviving successor fixes the rollout horizon and receipt
cap and selects only the signed policy-version graph.

The ignored raw development output is
`tmp/policy_dependency_sync/packet_debt_headroom_dev.json`.
