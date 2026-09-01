# Stochastic Markov-packet confirmation preregistration

Status: frozen design.  No seed in the registered namespace may be evaluated
before this document, the JSON configuration, runner and analyzer are committed.

## Scientific role

This is the independent CPU confirmation of the same clocked asynchronous MPG
mechanism studied in the exact-value stage.  It is the first stage with sampled
fixed-horizon Markov REINFORCE packets and complete transition-work accounting.
It is not a standard deep-MARL benchmark and cannot by itself make the project
ICML-ready.

The primary method is the maximal bounded-delay **single-flight certified
common step**.  The primary comparator is a fully utilized frozen-policy
barrier: every fast packet completed before the slowest agent returns is used
in that round's batch, and partial work cancelled at the barrier is charged.
A raw common-step asynchronous learner is a mandatory strong reference.  The
local-curvature and rate-balanced certificates are retained as ablations, not
as claimed improvements.

## Frozen population

- three-state persistent action-independent Markov chain;
- two distinct binary-softmax policy blocks and common reward;
- couplings `{0,.08,.16,.24}`;
- service ratios `{1,2,4,8}`, with `{2,4,8}` primary;
- fixed packet horizon `H=16`, batch size `16`;
- maximum elapsed time `180`, target normalized potential gap `.3`;
- one in-flight packet per asynchronous policy block;
- maximal registered step fraction `1.0` for every policy family;
- 64 fresh seeds in namespace
  `clocked-stochastic-confirmation-20260901-v1`.

There are 16 cells, five policies and 5,120 endpoint rows.  The seed namespace
is disjoint from every named development scan.  Development outcomes set no
confirmation threshold after this commit.

## Accounting

Each complete trajectory packet costs `batch_size * H = 256` joint Markov
transitions.  At a target-crossing time, unfinished in-flight work is charged
by elapsed service fraction.  At the terminal horizon, all partial in-flight
work is likewise charged.  In the barrier implementation, a packet that cannot
finish before the frozen-policy barrier is cancelled; its elapsed fraction is
charged as cancelled transition work.  Cancelled/partial work never becomes a
gradient update.

Service-duration RNGs and trajectory RNGs are separate.  Each agent and seed
uses the same named trajectory stream across policies; algorithms may consume
the stream at different joint policies and therefore this is variance
reduction, not pathwise identity.

## Mandatory prereproduction gates

1. `S1`: exact schema, expected row count, unique keys, finite endpoints and
   valid registered event delays.
2. `S2`: every primary cell has paired primary/barrier target coverage at least
   `0.95`.
3. `S3`: geometric mean of cell-median primary/barrier elapsed-time ratios is
   at most `0.75` over the 12 primary cells.
4. `S4`: corresponding fully charged transition-work ratio is at most `0.75`.
5. `S5`: the certified method is faster than the barrier in at least 10 of 12
   primary cells.
6. `S6`: its primary-cell final-gap geometric ratio to the barrier is at most
   `1.05`.
7. `S7`: relative to raw common-step asynchronous learning, the certified
   method's time and transition-work ratios are each at most `1.35`.  This is a
   certificate-cost bound, not a superiority claim.
8. `S8`: all transition accounting is nonnegative and internally ordered;
   target work is present exactly when target time is present.
9. `S9`: maximum realized event delay never exceeds the registered bound.
10. `S10`: the confirmation namespace is disjoint from named development.
11. `S11`: every policy has at least `0.95` aggregate target coverage.

Homogeneous service is descriptive and may show no benefit; no universal
no-harm claim is registered.  Raw asynchronous learning may be faster and that
fact must be reported.

## Reproducibility gate

After the first run finishes, rerun the unchanged committed source/config and
same seeds into a new directory.  `endpoints.jsonl` and the scientific summary
payload must be byte-identical.  This is `S12`.  Reproduction is performed even
if a scientific gate fails, so a negative result remains auditable.

Any failed mandatory gate stops the standard-MARL/GPU stage.  Gates, population,
seeds and methods must not be changed to rescue the result.

## Commands

```powershell
.\.venv\Scripts\python.exe -m experiments.clocked_async_mpg.run_stochastic_confirmation validate
.\.venv\Scripts\python.exe -m experiments.clocked_async_mpg.run_stochastic_confirmation run --workers 4 --output-dir <new-dir>
```

No HPC4, GPU or `/project` path is authorized by this preregistration.
