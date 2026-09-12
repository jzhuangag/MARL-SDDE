# Lyapunov parallel-commit CPU headroom validation

## Decision

**Stop this candidate.**  The frozen deterministic screen failed both oracle
headroom gates before sampled learning.  It must not be rescued by weakening
the comparator, selecting only favorable cells, retuning the declared grid, or
assigning a new experiment number to the same mechanism.

This result does not say that parallel commit control can never help.  It says
that the mechanism does not have the broad intrinsic value required by its
declared ICML story under the frozen exact-quadratic family and strong
per-scenario static comparators.

## Frozen provenance

- implementation/config commit: `68592f045999be0044ed8bfbb4c11657ede73a58`;
- branch: `codex/joint-ms-exp007c`;
- configuration SHA-256:
  `a141d7142dbf90612b4999e3b71491aab9449418782839d700100826eb13a0a3`;
- 192 deterministic scenarios: 4 construction seeds, 2 agent counts, 4
  interaction strengths, 2 anisotropies, and 3 service profiles;
- primary and isolated reproduction result SHA-256:
  `aba0ecbc52c66eb0c04c0038cf2d9093baafc3ba43f077e921c641fe87df2214`;
- targeted algebra/config tests: `15 passed`;
- full `clocked_async_mpg` regression in the `ust2` environment:
  `406 passed in 54.66s`.

The primary and reproduction JSON files were byte identical.  They remain
under the git-ignored local path
`experiments/clocked_async_mpg/results/lpc_headroom_20260903/`.

## Frozen comparison

For every scenario, the screen compares:

1. an unavailable arrival-time bound oracle;
2. the causal Lyapunov service/risk-debt controller;
3. the outcome-selected best fixed asynchronous scale in the declared grid;
4. the outcome-selected best sequential actor order.

The stronger of items 3 and 4 is the per-scenario static comparator.  The best
fixed asynchronous scale was stronger in all 192 scenarios.  This is stricter
than a single scale selected globally across all cells, but it was explicit in
the frozen implementation before any outcome was observed and is therefore
not reinterpreted after the fact.

## Results and gates

| Mandatory gate | Frozen threshold | Result | Pass |
|---|---:|---:|:---:|
| Oracle aggregate AUC gain | at least 5% | 1.0791% | no |
| Oracle cells with at least 5% gain | at least 60% | 11/192 = 5.7292% | no |
| Causal median oracle-headroom retention | at least 80% | 27.1419% | no |
| Causal cells with at least 5% gain | at least 60% | 0/192 | no |
| Service and risk debts finite and terminally nontrivial | required | service 1.5926; risk 0 | no |

The causal controller has geometric AUC ratio `1.04037545`, i.e. 4.0375%
worse than the strong static comparator, and mean scale `0.99426968`.

## Mechanism audit

Oracle value was narrow rather than broad:

| Slice | Oracle geometric ratio | Cells with at least 5% gain |
|---|---:|---:|
| 4 agents | 0.998473 | 1.04% |
| 6 agents | 0.980030 | 10.42% |
| interaction 0 | 1.000000 | 0% |
| interaction 0.25 | 1.000111 | 0% |
| interaction 0.75 | 0.992752 | 4.17% |
| interaction 1.5 | 0.964412 | 18.75% |
| balanced service | 0.977024 | 14.06% |
| two-tier service | 0.992344 | 3.12% |
| skewed service | 0.998380 | 0% |

Only 54/192 cells had any oracle improvement.  The largest gains occur in a
small high-interaction subset, chiefly six-agent, low-anisotropy, balanced or
two-tier cells.  This is a phase example, not a broad adaptive-learning story.

The causal failure has an additional identifiable cause.  Its service queue
uses unit arrivals for every ready proposal, so long-run queue feasibility
presses every scale toward one.  The observed mean scale of 0.9943 confirms
that the controller nearly becomes unrestricted simultaneous commit.  A
different service target could change this controller, but it cannot repair
the failed oracle ceiling and is therefore not evaluated on this frozen
candidate.

## Consequence for the paper program

No sampled confirmation, formal seeds, standard MARL benchmark, GPU, HPC4, or
remote storage work is authorized for this mechanism.  Its reusable pieces
are limited to the exact stale directional bound, the diagonal-plus-rank-one
box-QP solver, and the negative separation result showing that a Lyapunov
control layer is not valuable merely because asynchronous proposals exist.

The next research question must create an endogenous time-varying decision
whose oracle value survives strong static tuning.  It cannot be another
renaming of participation, commit scaling, or binary graph scheduling on a
stationary family.
