# Layer-0 HARL/MPE integration smoke

## Status

This is a non-scientific integration smoke.  It validates that the owned
clocked-packet interface can cross a standard multi-agent environment and a
standard neural policy implementation without violating its accounting or
ownership invariants.  It is not a benchmark comparison, a preregistered
experiment, a formal seed, or evidence of return improvement.

The upstream dependency is the public HARL checkout at commit
`b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`.  The checkout remains ignored and
unmodified; no upstream source is vendored into this repository.  The owned
adapter consists of:

- `harl_packet_overlay.py`: single-flight ownership, exact transition charging,
  categorical/diagonal-Gaussian policy drift, sample-split directional value,
  and the O(1) strategic-drift scaler;
- `run_harl_layer0_packet_smoke.py`: an end-to-end event-driven smoke using
  HARL's PettingZoo MPE wrapper and `StochasticPolicy` network;
- focused unit tests for the adapter and runner's deterministic utilities.

## Environment audit

An editable HARL installation was first attempted in the project's Python 3.11
environment.  Dependency resolution stopped before installation because the
pinned `pettingzoo==1.22.2` release excludes Python 3.11.  The main environment
was therefore not used or modified for this dependency.

The smoke used an isolated, ignored Python 3.9 environment:

| component | version |
|---|---:|
| Python | 3.9.2 |
| NumPy | 1.23.5 |
| PyTorch | 1.11.0+cpu |
| PettingZoo | 1.22.2 |
| SuperSuit | 3.7.0 |

This is a feasibility environment, not the final reproducibility lock.  Its
system SciPy installation declares NumPy `<1.23`, whereas the MPE/HARL smoke
needed NumPy 1.23.5.  The exercised paths completed, but the final benchmark
environment must be rebuilt from an explicit, compatible lock before GPU work.

## Upstream baseline smoke

Before adding the overlay, the unchanged HARL runner completed a minimal
64-environment-step CPU HAA2C run on continuous `simple_spread_v2` with three
non-shared actors, two rollout threads, episode length eight, and four learner
updates.  The run completed all updates at finite values.  Its terminal log
reported 143 FPS, average step reward `-4.2293`, and average episode reward
`-96.3874`.  These values only establish that the pinned upstream stack runs;
they are not scientific outcomes.

## Owned end-to-end packet smoke

The owned runner instantiated three distinct HARL `StochasticPolicy` blocks.
For each logical packet it collected one proposal trajectory and one
independent validation trajectory under the birth joint policy.  Both
trajectories were charged in full.  Each owner had at most one packet in
flight.  On completion, the runner checked exact self-freshness, measured
teammate policy drift on the validation observations, applied the closed-form
arrival-time scale, and relaunched the same owner when work remained.

The deterministic service law intentionally made the three agents
heterogeneous so that teammate blocks could change while a slower packet was
in flight.  The smoke used seed 1701, 12 packets, and eight environment steps
per proposal/validation trajectory.

| invariant or diagnostic | result |
|---|---:|
| packets launched/completed | 12 / 12 |
| charged environment steps | 192 |
| charged actor transitions | 576 |
| registry-completed actor transitions | 576 |
| maximum owner self-fresh error | 0 |
| packets with positive teammate drift | 7 |
| packets with positive update scale | 5 |
| per-agent parameter movement norms | 0.0500, 0.0822, 0.0500 |
| primary/reproduction SHA-256 | `d7f1cf8461d3809de0e545904178a0dbfc429c40e309efad8490375c6348cc76` |

The primary and isolated reproduction summaries are byte-identical.  Focused
adapter/runner tests passed 7/7; the complete `clocked_async_mpg` package passed
86/86 in the project environment.

The deterministic three-seed evaluation return changed from `-37.2537` to
`-37.2667`.  It is explicitly recorded as non-scientific and is not positive
performance evidence.  The short run has no comparator, no power, and no
frozen performance gate.

## What this closes and what remains open

The smoke closes the software-interface question: distinct policy ownership,
logical asynchronous completion, independent sample splitting, full resource
charging, observed teammate staleness, Lyapunov-debt scaling, and neural policy
updates coexist in an executable standard MPE stack.

It does not close the scientific benchmark question.  In particular:

1. the mean reference-batch KL is an observable drift diagnostic, not yet the
   theorem's required high-probability mixed-drift certificate;
2. curvature and mixed-drift constants are supplied upper bounds in this
   smoke, not learned or certified constants;
3. logical service time is simulated rather than produced by concurrent GPU
   workers;
4. the runner uses zero-baseline Monte Carlo policy gradients to align with the
   current theorem, not the full HAA2C/HAPPO training stack;
5. no return, wall-clock, sample-efficiency, or no-harm claim follows from this
   run.

The next admissible step is therefore an outcome-free benchmark
preregistration and environment lock, followed by a small GPU pilot comparing
the same charged packet mechanism with strong synchronous and raw-asynchronous
baselines.  The pilot must be stopped if static feasibility, accounting,
nontriviality, or reproducibility gates fail.

## Provenance

| item | SHA-256 |
|---|---|
| owned overlay | `03f5b3d4e48e8f53641e5c506a03e577e47f6d282628458c052b115f0610b2ee` |
| end-to-end runner | `a0a3b3c6d127aa8c4ccf51006e9639079504892cc3b4fbfe6ac4dab212881d23` |
| overlay tests | `586c6afadd39743e08a7a6faba120e2958b175089292bd98a1aef8d16b5a167a` |
| runner tests | `bfd24a7ae316150807096ce81634523a3158156caa1409c49194869bfae74e2e` |
| upstream HAA2C config | `30f87c99a7ed2a2ddc6f898cf7e23d4dc34cabcee6e9e579ff3dbadbe7267e1e` |
| upstream setup metadata | `111c3b00ef6810aa186eae4889855739188511f02411d9bd0b81e0d5ccc3cb62` |

The HARL bibliographic and source metadata used by the project were already
verified in `clocked_async_mpg_citation_verification.json`.
