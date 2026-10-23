# TSP-V2 learning-progress sensor G0-B preregistration

## Material Passport

- Experiment: `TSP-V2-MARL-PROGRESS-G0B`
- Role: outcome-isolated interface and sensor qualification
- Status: frozen before HPC4 execution
- Scientific output: no manuscript performance claim

## Rationale

G0-A1 established that the scientific payload runs on MaMuJoCo but also showed
that its common runtime omitted the dependency required to construct the
PettingZoo environment.
The four completed MaMuJoCo cells are retained as incomplete diagnostic data
and are not reused in G0-B.
G0-B uses two new seeds for every task and binds each task to an existing,
previously exercised runtime before any outcome is generated.

The scientific question is unchanged: does a short, fully charged
micro-training branch expose a finite and reproducible learning-progress
signal for every `q in {1,2,4,8}` on PettingZoo MPE and MaMuJoCo?
Every branch starts from byte-identical actor and critic parameters, performs
24 learning updates, and records only rewards observed during those updates.
No deterministic evaluation return or manuscript metric is computed.

## Frozen cells and runtimes

- PettingZoo `simple_spread_v2`, independent and shared coupling, seeds
  `140201` and `140202`.
- MaMuJoCo `HalfCheetah-v2/2x3`, independent and shared coupling, seeds
  `140201` and `140202`.
- Candidates `q={1,2,4,8}` and 24 updates per candidate.
- PettingZoo uses the preserved TSP confirmation Python environment and HARL
  checkout at commit `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`.
- MaMuJoCo uses the preserved Two-Clocks Python environment and HARL checkout
  that completed all four G0-A1 MaMuJoCo tasks.
- User-site packages are disabled for both runtimes.

The runtime split is task-specific dependency isolation, not an algorithmic
treatment: the same owned runner, architecture, optimizer settings, candidate
set, charging rules, statistic, and analyzer are used in both environments.

## Mandatory qualification gates

1. All eight registered cells complete exactly once.
2. Every candidate branch starts from the same parameter SHA-256 within a cell.
3. Every reward trace contains exactly 24 finite entries and no evaluation
   return.
4. Message and environment charges exactly match the public formulas.
5. Independent MaMuJoCo selects `q=8` in at least one of two seeds and does not
   select `q=1` in both seeds.
6. Shared MaMuJoCo selects an interior action `q in {2,4}` in at least one of
   two seeds.
7. PettingZoo produces finite, nonconstant progress traces in both regimes.
8. Two clean analyzer executions are byte-identical.

Failure of a payload or mandatory gate stops this sensor design without retry.
Passing G0-B authorizes design of the coefficient-certificate layer only; it
does not authorize manuscript inclusion, formal seeds, or a return claim.

