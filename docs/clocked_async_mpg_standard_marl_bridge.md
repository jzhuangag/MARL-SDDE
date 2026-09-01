# Standard-MARL bridge for Clocked Policy Optimization

Status: outcome-free design.  No standard-task seed, return, win rate, or GPU
result has been generated.  This document fixes the intended scientific
comparison before implementation and preregistration.

## 1. Unified question

The project asks one question:

> When heterogeneous learners own distinct interacting policy blocks, how
> should stochastic policy updates be scheduled and scaled so that removing a
> rollout barrier yields wall-clock learning value without ignoring strategic
> policy staleness?

The answer is **Clocked Policy Optimization (CPO)**:

1. each actor block has at most one rollout/update packet in flight;
2. self-freshness removes own-block staleness;
3. teammate-policy changes create interaction-weighted strategic drift;
4. a pathwise Lyapunov--Krasovskii condition controls arbitrary bounded
   completion order;
5. a scalar Lyapunov-debt rule scales practical proposals online.

This is centralized asynchronous training with decentralized execution.  A
training parameter server stores actor snapshots and, where used, a centralized
critic.  At execution, every agent runs only its local actor and the proposed
method adds no communication protocol.

## 2. Relation to adjacent work

- Independent policy gradient in Markov potential games proves convergence for
  policies updated in tandem; it does not model wall-clock completion order or
  teammate-policy packet staleness.
- HAPPO/HARL handles distinct heterogeneous actors through coordinated
  sequential updates and supplies strong synchronous baselines.
- MAPPO is a mandatory strong cooperative-MARL baseline, not evidence that a
  weaker asynchronous baseline is sufficient.
- AFedPG supplies an asynchronous wall-clock result for many workers training
  one shared global policy.  CPO instead has strategically different policy
  blocks: self-freshness and cross-policy drift are the central quantities.
- SMACv2 is preferred over legacy SMAC for the discrete benchmark because its
  procedural variation and partial observability require closed-loop policies.

The claim must never be "the first asynchronous RL algorithm."  The narrow
novelty target is a pathwise finite-time wall-clock and strategic-drift theory
for distinct interacting Markov-game policy blocks, plus a low-complexity
online realization.

## 3. Benchmark layers

### Layer 0: local CPU integration smoke

Use PettingZoo MPE `simple_spread_v2` only to validate shapes, non-shared actor
ownership, packet snapshots, KL drift, transition accounting, deterministic
replay and process teardown.  It is not a headline scientific result.

### Layer 1: continuous heterogeneous cooperative control

Use the official HARL configurations as the starting point for:

- MAMuJoCo `Walker2d-v2`, `6x1`;
- MAMuJoCo `Ant-v2`, `4x2`.

These tasks expose distinct continuous-action policy blocks and have official
HAA2C, HAPPO and MAPPO configurations.  A preflight may reduce the task set only
for installation or deterministic-crash reasons recorded before any pilot
outcome; it may not select tasks by observed method advantage.

### Layer 2: stochastic partially observable cooperation

Use SMACv2:

- `terran_5_vs_5`;
- `zerg_10_vs_10`.

This layer tests whether the mechanism survives recurrent policies, stochastic
unit composition and partial observation.  It is not covered by the tabular
MPG theorem and must be labeled an empirical extension.

## 4. Algorithm and baselines

The protocol-matched family uses non-shared actors and identical packet work:

1. fully utilized frozen-policy barrier;
2. raw single-flight async with the same base optimizer;
3. delay-only scaled async;
4. pathwise conservative async;
5. sample-split strategic-drift CPO;
6. oracle directional-value CPO, development/diagnostic only.

External strong baselines use their official synchronous training semantics:

7. HAA2C;
8. HAPPO;
9. MAPPO without actor parameter sharing where the task permits.

The primary theory-aligned implementation uses Monte Carlo actor returns with
an action-independent critic baseline frozen inside each packet.  A PPO/HAPPO
variant is a practical secondary implementation; clipping, Adam and a learned
bootstrapped critic are outside the current finite-time theorem and must not be
silently claimed as covered.

Official HARL commit inspected for the design:

`b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`

The upstream checkout contains no root license file in that commit.  Therefore
the repository will not vendor or redistribute HARL source.  HPC runs must use
a pinned external checkout plus a small auditable overlay owned by this
project, while retaining upstream provenance.

## 5. What counts as heterogeneous service

Artificial `sleep` is not sufficient main evidence.  The pilot must report two
profiles:

1. **endogenous:** measured actor-packet time under the task's distinct policy
   blocks and declared CPU/GPU allocation;
2. **controlled:** fixed per-actor packet workloads or CPU affinity that perform
   real, fully charged rollout/optimization work.

Every method receives the same actor-specific packet workload.  Logical event
time, actual elapsed time, environment transitions, optimizer FLOP proxy,
cancelled partial work and GPU utilization are all reported.  A shared-buffer
official baseline may reuse a joint rollout across actors; that efficiency is
retained rather than hidden, even though it is not protocol matched.

## 6. Metrics and nontrivial gates

Primary performance objects:

- actual time and charged work to fixed, task-public return thresholds;
- interquartile mean return versus actual time and versus transitions;
- final return and lower-tail return across seeds;
- policy-lag events and teammate KL drift at arrival;
- proposal scale, rejection rate, certificate penalty and normalized debt;
- CPU/GPU utilization and actor idle fraction.

A pilot can authorize formal seeds only if all mandatory conditions hold:

1. finite runs, exact work identities and no process leaks;
2. CPO beats the fully utilized barrier in time-to-threshold on both task
   families without worse transition work by more than a frozen tolerance;
3. CPO improves materially over the conservative pathwise implementation;
4. final return is noninferior to the protocol-matched raw async comparator;
5. at least one strategic-drift ablation (`S=0`) is worse in high-drift cells;
6. the controller is nontrivial: scales vary with measured teammate drift and
   it does not reduce to permanent maximum scale or permanent rejection;
7. official HAPPO/MAPPO learning curves remain visible as strong external
   references;
8. all seeds, thresholds, analysis code and source hashes are frozen before the
   pilot and replaced by a fresh registry before formal confirmation.

Exact numerical thresholds and seeds belong in a later preregistration commit,
after environment smoke tests and resource estimates.  This design does not
authorize a GPU job by itself.

## 7. Remaining pre-GPU work

1. implement the external-checkout overlay without copying upstream source;
2. run Layer-0 CPU smoke and transition-accounting tests;
3. specify the predictable-noise theorem for the sample-split scale or clearly
   freeze it as an expectation-level practical extension;
4. measure one-task smoke resource use and then freeze task budgets, pilot
   seeds, gates and Slurm layout;
5. only then submit the minimal pilot to HPC4 scratch, with no `/project`
   output until validation passes.
