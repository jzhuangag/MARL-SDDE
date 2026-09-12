# Two Clocks standard-environment pilot validation

## Decision

The frozen pilot **failed** and formal escalation is stopped.  The failure is
not caused by missing work or insufficient asynchronous depth: under
heterogeneous service, immediate async produced `2.75x` the Ant barrier update
depth and `2.60x` the SMACv2 barrier depth while using exactly the same fully
charged packet and transition counts.  Nevertheless, return degraded.

The frozen analyzer passed 5 of 12 mandatory gates: P1, P2, P9, P11, and P12.
It failed P3--P8 and P10.  A supplemental semantic audit also found that the
SMACv2 initial evaluations were not paired despite identical registered seeds;
therefore the numerical P1 pass means only that the implemented accounting
checks passed, not that the intended paired-comparison contract was achieved.
This makes the stop decision strictly stronger.

No formal seed was created or run.  Thresholds, seeds, tasks, methods, service
profiles, and outcomes were not changed.

## Execution and operational amendment

Preregistration commit:
`b3a99190f6e6ec345e49ec63f88fcc9c2e0ee758`.

The first four jobs (`1810507`--`1810510`) stopped before their first
scientific row because deterministic PyTorch required
`CUBLAS_WORKSPACE_CONFIG`.  All exited `1:0` after 57--77 seconds and none
wrote `summary.json`.  Operational amendment 1 added only
`CUBLAS_WORKSPACE_CONFIG=:4096:8`; no scientific design field changed.  The
amended execution commit was
`a0786824972d8203ad803970b586eebda9665105`.

Valid jobs were:

| Job | Task | Run | Node | Elapsed | State |
|---:|---|---|---|---:|---|
| 1810518 | MAMuJoCo Ant 4x2 | primary | gpu12 | 03:25 | COMPLETED 0:0 |
| 1810519 | MAMuJoCo Ant 4x2 | reproduction | gpu12 | 03:25 | COMPLETED 0:0 |
| 1810520 | SMACv2 Terran 5v5 | primary | gpu13 | 22:06 | COMPLETED 0:0 |
| 1810521 | SMACv2 Terran 5v5 | reproduction | gpu13 | 23:03 | COMPLETED 0:0 |

Every per-run `SHA256SUMS` manifest passed.  The valid result tree occupies
about 1.4 MB under
`/scratch/jzhuangag/MARL-SDDE-TwoClocks-20260902`; no result was written to
`/project` or `/home`.

## Frozen outcome

The primary heterogeneous-service result was:

| Metric | Aggregate / Ant / SMACv2 |
|---|---:|
| async vs frozen-barrier logical-time AUC | `-7.7367%` / `-3.1453%` / `-12.3281%` |
| cells with positive async-vs-barrier AUC | `3/8` |
| async vs delay-scaled AUC | `-19.6910%` aggregate; positive in `1/8` cells |
| async vs barrier terminal return | Ant `-13.8614%`; SMACv2 `-33.1493%` |
| async terminal-minus-initial return | Ant `-2.5051`; SMACv2 `-0.3822` |
| mean clipping fraction | `0.09971` |

Taskwise balanced-service async-versus-barrier AUC gains were `-0.7515%` for
Ant and `-29.5582%` for SMACv2.  The required heterogeneous-over-balanced
phase ordering therefore did not hold in both tasks.

The exact frozen gate ledger is:

| Gate | Result |
|---|---:|
| P1 implemented validity/equal-work checks | pass |
| P2 adaptive-depth ratio | pass |
| P3 heterogeneous AUC gain | **fail** |
| P4 positive cells | **fail** |
| P5 delay-scaled comparison | **fail** |
| P6 heterogeneous-over-balanced phase | **fail** |
| P7 terminal safety | **fail** |
| P8 positive learning change | **fail** |
| P9 clipping | pass |
| P10 byte reproduction | **fail** |
| P11 provenance | pass |
| P12 pilot/formal separation | pass |

## Reproduction and pairing audit

Ant primary and reproduction summaries were byte-identical, both with SHA-256
`21194f78e79b08928d660e2d870d291af93af576386c5c36377f877f14ddaeff`.

SMACv2 was not byte reproducible:

- primary:
  `a0b8d29b56c9edd4ddf0620bc4f785f3bd222a3c51bb9d8f433064d2a9fd59ef`;
- reproduction:
  `1329f8409297f2ffd71ff12596d868f46d7388c85a81e92233cc729ae0a5c6bf`.

More importantly, evaluation occurs before any method-specific update, so its
initial return should be invariant across the six method/profile rows for a
paired task seed.  Ant's maximum within-seed range was exactly zero in both
runs.  SMACv2's ranges were `0.607--1.740` in the primary run and
`1.330--3.035` in reproduction.  The public SC2 random seed was supplied, but
the observed trajectory was not deterministic enough to provide the intended
common-random-number pairing.  The four-seed SMAC contrasts must therefore be
treated as noisy descriptive values, not paired evidence.

The frozen machine analyzer did not include this initial-invariance check, so
its P1 field remains preserved as originally computed.  This report does not
retroactively edit that output; it records the stricter audit separately.

## Scientific interpretation

This pilot rejects the simple proposition that more self-fresh updates are by
themselves beneficial.  Immediate async used a fixed per-packet step, so its
larger adaptive depth also created larger accumulated optimization motion.
The barrier averaged same-birth gradients and reduced estimator variance;
the delay-scaled method attenuated strategically stale packets.  Both effects
can dominate freshness.  Hence the pilot confounds neither missing data nor an
idle barrier: it exposes the missing control variable.

The next viable unified question is therefore not “async or barrier?” but:

> how should an asynchronous MARL learner use a Lyapunov drift budget to
> jointly admit and scale each self-fresh but strategically stale packet, so
> that additional adaptive depth is used only when its certified descent value
> exceeds its staleness and variance debt?

This moves Lyapunov drift from an after-the-fact analysis tool into the online
decision rule.  Before another standard GPU pilot, a CPU theory/mechanism gate
must establish a performance bound and nontrivial oracle headroom against both
the strong frozen barrier and delay scaling.  It must also use a benchmark
protocol with verified paired initial trajectories or an analysis explicitly
designed for independent stochastic replications.  No new GPU experiment is
authorized by the present result.

## Provenance

Frozen analyzer output:
`/scratch/jzhuangag/MARL-SDDE-TwoClocks-20260902/results/standard-pilot-a0786824972d8203ad803970b586eebda9665105/analysis/validation.json`.

Its SHA-256 is
`0678ec1fae6768978447efc27e14e71b21ce91359a66d3c192f5162e4eed7334`.

