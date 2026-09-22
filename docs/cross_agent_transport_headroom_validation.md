# Cross-agent transport outcome-free headroom validation

## Material Passport

- Artifact ID: `CAGT-HEADROOM-STATIC-20260901-RESULT`
- Parent preregistration commit: `eb56fb5`
- Artifact type: deterministic, outcome-free design-feasibility scan
- Scientific efficacy population: none
- Formal evidence: none
- GPU/HPC4 execution: none
- Raw result: `docs/cross_agent_transport_headroom_results.json` (local only)
- Raw bytes: `5,201,673`
- Raw SHA-256: `22481e794219f09f1ebf2dee9bbd204a092860c90fc9d2e266dd2009898930ea`

## Frozen result

The scan completed all 17,820 declared trajectories over 324 cells and 55
policies.  All reported regret and terminal-gradient values are finite.  Exact
quadratic transport never took a potential-decreasing coordinate step.

| Phase | Cells | Geometric transport / strong-envelope regret | Cells with at least 10% gain |
|---|---:|---:|---:|
| low | 54 | 1.233265 | 12.96% |
| transition | 216 | 1.082930 | 25.46% |
| high | 54 | 1.028479 | 27.78% |

In the high phase with the fully charged unit HVP overhead, the geometric
ratio is `1.352359`.  The full frozen gate ledger is therefore:

| Gate | Result |
|---|---:|
| all trajectories complete and finite | pass |
| high geometric ratio at most 0.90 | **fail** |
| at least 60% high cells obtain at least 10% gain | **fail** |
| transition geometric ratio at most 0.95 | **fail** |
| low geometric ratio at most 1.05 | **fail** |
| high, full-HVP-cost ratio at most 0.95 | **fail** |
| no harmful exact-transport coordinate step | pass |

Only 2/7 mandatory gates pass.  `all_gates_pass=false` and
`efficacy_pilot_authorized=false`.

## Diagnostic decomposition

This decomposition was not a frozen success gate and is reported only to
explain the stop decision.  Across all phases, the geometric regret ratio is
`0.860767` at zero HVP overhead, `1.007713` at overhead 0.25, and `1.522664`
at overhead 1.0.  Thus the Taylor correction is useful when it is effectively
free, but unconditional correction does not retain enough wall-clock value
after realistic compute charging.  The strong-envelope winner is most often
fresh serial (147 cells) or a cross-debt gate (117 cells), with raw async,
barrier, delay-adaptive and age-decay rules accounting for the remainder.

This does not refute the transport-radius lemma or the conditional event-time
bound.  It refutes the current paper route in which every completion receives
one fully charged cross-agent HVP.  A proof that such a method converges would
not repair its missing performance headroom.

## Decision

Permanently stop the current **unconditional full-transport performance
route**.  Do not launch its stochastic CPU pilot, formal seeds, standard MARL
benchmark, GPU job, or wall-clock separation claim.  Do not modify the frozen
phase labels, comparator envelope, overhead grid, gates, or result.

Any successor must pose a different unified stochastic-control problem in
which correction compute is itself an online action and is fully charged.  In
particular, a compute-aware controller may choose among applying a cheap raw
proposal, transporting it, refreshing it, or rejecting it under a real
correction-compute budget.  Such a successor needs a new outcome-free theorem
interface and a new static headroom gate; the zero-overhead slice above cannot
serve as efficacy evidence or as authorization.

