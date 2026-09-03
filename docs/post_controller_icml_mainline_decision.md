# ICML mainline decision after controller falsification

Date: 2026-09-03.

## Decision

Retain **Two Clocks, One Game** as the sole surviving ICML candidate.  Stop
inventing online participation, graph, refresh, trust-radius, commit-scale, or
dual-repricing controllers on stationary learning families.

The unified paper question is:

> In CTDE with distinct asynchronously trained policy blocks, when does
> removing the global learner barrier produce genuine learning progress, and
> when does off-diagonal strategic staleness erase that gain?

This is a rate--coupling phase theorem with a low-complexity learning rule, not
a claim that one adaptive controller universally dominates every tuned static
algorithm.

## One contribution chain

1. **Structural identity.**  Under single-flight ownership, an arriving packet
   is fresh in its owner's policy block and stale only through teammate-policy
   motion.
2. **Lyapunov design and upper bound.**  An off-diagonal
   Lyapunov--Krasovskii history functional selects a certified common or
   agent-wise step and yields a finite-time Markov-packet stationarity bound.
3. **Phase/lower side.**  Heterogeneous service creates adaptive-query depth;
   interaction-weighted motion destroys uniformly useful stale progress; each
   strategically essential slow actor remains a clock bottleneck.
4. **Experiments.**  Exact, stochastic finite-game, and standard neural tasks
   must exhibit the same favorable/unfavorable phase under complete work
   accounting.

Lyapunov is a design tool because its off-diagonal history weights determine
the admissible step domain.  The online actor update remains ordinary
`O(dim(theta_i))`; no subset scan, QP, covariance inverse, or dense
preconditioner is required.  SDDE is optional interpretation only.

## What the evidence currently says

Positive theorem-facing evidence already exists:

- exact multi-state confirmation: geometric asynchronous/barrier target-time
  ratio `0.444557`, all 12 heterogeneous cells faster;
- independent fully charged stochastic Markov packets: wall-clock ratio
  `0.431353`, charged-transition ratio `0.429687`, all 12 heterogeneous cells
  faster;
- homogeneous/high-coupling controls include cases where the barrier wins, as
  required by a phase claim rather than universal dominance.

The first standard GPU pilot failed.  Its immediate asynchronous method used a
fixed `5e-4` Monte-Carlo policy-gradient step with no critic.  Under
heterogeneous service it executed 2.75 times as many Ant updates and 2.60 times
as many SMACv2 updates as the barrier, while applying the same per-update step
cap.  Both tasks had negative async learning change.  Thus the pilot tested
raw update accumulation, not a mature actor--critic instantiation of the
off-diagonal Lyapunov design.  Its negative result and SMAC pairing defect
remain immutable.

The subsequent online-control attempts cannot repair this gap: their oracle
or causal headroom screens failed.  They remain stopped.

## Final CPU bridge before another GPU request

Build one new standard MPE bridge with the pinned HARL implementation, using
two public task families (`simple_spread_v2` and `simple_reference_v2`) and
distinct policy blocks.  It must use an action-independent learned/frozen
value baseline or the upstream HAA2C/HAPPO advantage interface; another
zero-baseline Monte-Carlo experiment is not informative.

The comparison must contain:

1. single-flight off-diagonal-step asynchronous learning;
2. raw asynchronous learning with the same estimator and charged work;
3. a generic full-staleness/delay-scaled asynchronous reference;
4. a fully utilized frozen-policy barrier;
5. the upstream synchronous HAA2C/HAPPO learning curve as an external quality
   anchor.

Training transitions, cancelled/partial work, optimizer updates, cumulative
KL/motion, logical service time, physical runtime, and evaluation transitions
must be separate.  In particular, the methods must either have matched
cumulative trust-region motion or report it explicitly; equal packet count is
not enough when the barrier averages multiple same-policy packets.

The first implementation stage is outcome-free: validate buffers, advantage
freezing, owner self-freshness, barrier averaging, exact charging, paired
initial evaluation, and deterministic replay.  Only then freeze fresh CPU
pilot seeds and gates.

## CPU survival criteria

The MPE bridge must show, before any new GPU work:

- positive learning change for every primary method/task;
- at least 5% heterogeneous-service AUC gain over the fully utilized barrier
  in aggregate and positive taskwise gain on both task families;
- at least 60% paired-cell directionality;
- no worse than 2% return/AUC loss to the stronger raw or generic asynchronous
  reference;
- a favorable heterogeneous-over-balanced ordering on both tasks;
- exact accounting and byte-identical deterministic components;
- no reliance on an outcome-selected interaction proxy or step.

Failure stops the standard-neural bridge and leaves a theory/synthetic paper,
not an ICML-complete empirical package.  Passing authorizes a separate GPU
preregistration; it does not convert the CPU pilot into formal evidence.

## Current readiness

The frozen public-MPE CPU bridge has now failed its preregistered survival
criteria.  It produced only `0.492%` aggregate normalized heterogeneous AUC
gain versus the frozen barrier (required `5%`), with positive direction on
`simple_reference_v2` and negative direction on `simple_spread_v2`; positive
learning and taskwise phase-order gates also failed.  Exact reproduction
passed.  The upstream HAA2C anchor and GPU escalation are therefore stopped.

The theory-facing finite-game story remains coherent and has positive
reproducible evidence, but this particular neural bridge does not provide the
cross-task external evidence required for an ICML-complete package.  Another
controller or outcome-guided retuning of this experiment is forbidden.  Any
future ICML attempt must first change the problem-level source of value or
derive a mature actor--critic algorithm before assigning a new experiment,
rather than treating the present failure as an implementation parameter to
tune away.
