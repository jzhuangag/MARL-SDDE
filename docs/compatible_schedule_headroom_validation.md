# Compatible asynchronous-update scheduling headroom validation

## Decision

**Stop this scheduling candidate before stochastic learning or GPU work.**

The frozen deterministic ceiling does establish that compatible concurrent
updates can process substantially more ready proposals than a sequential
actor-update rule.  It does **not** establish the paper-critical claim that a
Lyapunov/MaxWeight scheduler materially improves over strong compatible
coloring.  Three of six mandatory gates fail, so no causal-cone theorem program,
potential-game pilot, formal seed registry, standard-RL benchmark or GPU run is
authorized by this result.

This decision does not alter any earlier experiment, gate or conclusion.  The
compatible-update smoothness inequality, independent-set algebra, path MWIS
solver and queue-drift identity remain valid structural tools.  They are not an
efficacy result and do not by themselves support an ICML contribution.

## Material Passport

- Artifact ID: `CAUS-HEADROOM-STATIC-20260901`
- Evidence class: outcome-free deterministic queue/scheduling ceiling
- Frozen design commit: `cc09594`
- Parent decision commit: `31b74e8`
- Design hash: `06eba45b60bb1ba0b7bd7e2c43152bc09cad1910e87128ad08b2c6fd72aa1cac`
- Primary result: `compatible_schedule_headroom_results.json`
- Primary result bytes: `107213`
- Primary result SHA-256: `745F4ABEFC9BA162C0CA4C205BE3B850D319A8ADBA24214C1D456A72AA78BA00`
- Scientific trajectories, returns and gradient samples: none
- Random or formal seeds: none
- GPU/HPC4/remote storage: not used
- Python: `3.11.13`

## Frozen execution

The design contains 48 scenarios and 5 policies, hence 240 complete runs.  It
crosses 16/36/64 actors, path/tree/grid/clustered conflict graphs and four
deterministic readiness workloads over 240 epochs.  Every policy receives the
same arrivals and progress weights.  The per-scenario strong baseline is the
best total-cost member of best-ready-color, cyclic-color and static-priority
compatible scheduling.  The metric is queue area plus 240 times terminal
backlog.

The primary command was:

```powershell
.\.venv\Scripts\python.exe -m experiments.policy_update_backpressure.compatible_schedule_headroom run --output docs/compatible_schedule_headroom_results.json
```

A clean rerun to a temporary path produced the identical 107,213-byte file and
identical SHA-256.  The temporary reproduction was then removed.

Post-result verification used the frozen source without changing the controller
or gates:

- compatible-update targeted tests: `11 passed in 0.54s`;
- complete experiment regression: `924 passed, 7 skipped in 123.17s`.

## Mandatory gates

| Frozen gate | Threshold | Observed | Result |
|---|---:|---:|---:|
| Complete, finite and compatible | all 240 runs | all 240 | pass |
| Geometric dynamic/strong cost ratio | at most 0.85 | **1.0267566** | **fail** |
| Scenarios with at least 10% cost reduction | at least 60% | **18.75%** | **fail** |
| Geometric throughput ratio versus sequential | at least 2.0 | **7.6147004** | pass |
| Full dynamic actor coverage | 100% | 100% | pass |
| Dynamic no worse than strong baseline | every cell | **20/48 worse** | **fail** |

The median cost reduction is exactly 0%.  Dynamic scheduling improves 22
scenarios, ties 6 and worsens 20.  The best cost ratio is 0.4777448; the worst
is 2.0642458.  The throughput result therefore cannot rescue the candidate:
parallel compatibility is useful relative to serial execution, but the
proposed dynamic scheduler has no robust advantage over already-compatible
strong schedules.

## Prespecified-population diagnostics

These are diagnostics of the frozen population, not replacement gates.

| Slice | Geometric cost ratio | Fraction at least 10% better | Worse cells |
|---|---:|---:|---:|
| path | 0.943664 | 25.00% | 0/12 |
| tree | 1.036560 | 16.67% | 7/12 |
| grid | 1.304543 | 0.00% | 9/12 |
| clustered | 0.870963 | 33.33% | 4/12 |
| homogeneous | 1.038109 | 8.33% | 5/12 |
| two-tier | 0.997292 | 16.67% | 6/12 |
| rotating burst | 1.118841 | 25.00% | 6/12 |
| color skew | 0.959481 | 25.00% | 3/12 |
| 16 actors | 1.072934 | 6.25% | 8/16 |
| 36 actors | 0.979343 | 18.75% | 6/16 |
| 64 actors | 1.030136 | 31.25% | 6/16 |

Static priority is the hindsight strong baseline in 37/48 scenarios, cyclic
color in 6 and best ready color in 5.  The failure is therefore not a single
baseline artifact.  Grid scenarios are a particularly clear counterexample to
the claimed broad dynamic value.

## Interpretation and stop boundary

The positive 7.61x serial-throughput comparison is an expected consequence of
serving independent sets and is not sufficient novelty or evidence for an
ICML main claim.  Against the correct strong compatible baseline, the dynamic
term sometimes helps but is neither material in aggregate nor uniformly safe.
The frozen result rejects this concrete `compatible_maxweight` construction and
its claimed scheduling advantage; it is not a theorem that all asynchronous
MARL scheduling is impossible.

Do not change the arrival traces, potential weight, graphs, horizon, strong
baseline, cost or gates and rerun this identifier.  Do not launch the planned
stochastic potential-game pilot.  A future asynchronous-MARL direction would
need a different endogenous learning constraint or information structure—not
another weight, coloring or workload sweep—and must establish paper-level
novelty and oracle value before experiments.

## Source provenance

| File | SHA-256 |
|---|---|
| `compatible_schedule_headroom_plan.md` | `D004DFDED12658E55DF4A4ECF921AAF6032AB8E61ADF86BB2D0C2BB2822154BC` |
| `compatible_schedule_headroom_manifest.json` | `9392C1D4BD6BADDDCFB04E04F74934503697EB01B6FC0B3B5E17F46712439B2E` |
| `compatible_schedule_headroom.py` | `D667F4673A40C5513159DA08B94978B16E05E01384F8A922CD1FC582073172F3` |
| `compatible_update_theory.py` | `45F4B0CEE21CDAF4E14404CD03D01629DF084C301DEAB669A0E2B3E10CEFB268` |
| `test_compatible_schedule_headroom.py` | `0B33E0A65B34AD7821AC64E0936416428EC48BEF015B0FD9719186474576AE02` |
| `test_compatible_update_theory.py` | `5AB413A08AC8744C50EFDB96511FDCAA442621F7F53AB1F5E53D775AFFAC35D9` |
