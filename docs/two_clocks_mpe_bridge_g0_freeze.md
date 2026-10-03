# Two Clocks public-MPE bridge: outcome-free G0 freeze

Date: 2026-09-03.

## Scope

This commit freezes an **outcome-free implementation gate**, not a scientific
pilot.  The gate exercises pinned HARL actors on public PettingZoo MPE
`simple_spread_v2` and `simple_reference_v2`.  It must not emit reward, return,
success, advantage, gradient, or loss fields, and it cannot authorize a GPU
run or support a learning-performance claim.

The compared execution contracts are fixed before G0 execution:

1. single-flight asynchronous learning with the off-diagonal
   Lyapunov--Krasovskii scale;
2. raw single-flight asynchronous learning;
3. full-event-delay-scaled asynchronous learning;
4. a fully utilized frozen-policy barrier that averages all complete packets
   in a round and charges unfinished tail work.

Every complete packet contains two independent, fully charged trajectories.
Trajectory seeds depend on task, owner, owner-local packet index, and replicate,
but not on method.  Initial policies and the past-data frozen control variate
must be byte-paired across methods and service profiles.

## Lyapunov design interface

For the G0 neural adapter, the declared block sensitivity matrix has diagonal
`5.0` and off-diagonal entries `0.5`.  The pathwise single-flight certificate
uses maximum event delay `8`.  If its common finite-policy step is
`alpha_LK`, the applied neural multiplier is

`alpha_LK / (1 / max_i L_ii)`.

Thus the off-diagonal interaction history changes the actual update applied by
the candidate method.  This remains an empirical neural interface: G0 does
not assert that an unconstrained neural actor satisfies the global finite-game
smoothness constants.

## Frozen provenance

- HARL commit: `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`
- contract SHA-256:
  `cef6f530196d1351567658649663a361dead7f6c19c6a8299eb294c4f4d048d7`
- runner SHA-256:
  `d1764fa7b3e81e7d3a112e07f3839900b6f07361e5a9bf84c89036b37c136ae8`
- configuration SHA-256:
  `75aaeeb197abbb1805e92d24efc16100002479d5d313d56f94a0b20081e6e562`
- targeted static tests: `12 passed`

## Mandatory G0 conditions

G0 passes only if all registered task shapes match; the pinned checkout is
clean; every case has exact service/work accounting; every asynchronous owner
is self-fresh and every barrier round is policy-frozen; all complete packets
are applied or legitimately batched; partial tail work is visible; initial
policies and frozen control variates are paired; policy-motion diagnostics are
finite and nonzero; and no prohibited outcome field is emitted.

Failure stops the bridge.  Passing G0 permits a separate pilot
preregistration; it does not permit reading G0 internals to select a task,
learning rate, threshold, or comparator.
