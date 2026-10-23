# Standard-MARL bridge for Clocked Policy Optimization

Status: outcome-free design for the Two Clocks mainline. No standard-task seed,
return, win rate, or GPU result has been generated. This document fixes the
intended comparison before implementation and preregistration.

## 1. Unified question and primary algorithm

The project asks:

> When heterogeneous learners own distinct interacting policy blocks, can a
> barrier-free learner obtain wall-clock value while certifying the strategic
> staleness created by teammate-policy changes?

The primary theorem algorithm is the **single-flight pathwise-certified
common-step learner** on the finite-policy class:

1. each non-shared actor block has at most one fixed-work rollout/update packet
   in flight;
2. a returned packet is self-fresh in its owner block and strategically stale
   only through teammate-block motion;
3. it is applied immediately with one predeclared common step certified from
   the interaction envelope and bounded completion window;
4. the owner then launches its next packet from the current joint-policy
   version.

The online update is ordinary block policy gradient. The pathwise
Lyapunov--Krasovskii certificate is the core mechanism; it is not an online
strategic-debt controller, a sample-split scale-selection procedure, or an
oracle directional-value method.

The unconstrained HARL neural actor does not currently instantiate the uniform
constants required by that certificate.  In the standard benchmark the same
single-flight/common-step timing rule is therefore an explicitly empirical
neural extension, not a theorem-covered neural algorithm.

This remains centralized training with decentralized execution. Training may
use a parameter server for actor snapshots and, where used, a centralized
critic. During execution every agent uses only its local actor, with no added
communication protocol.

## 2. Relation to adjacent work

- Independent policy gradient in Markov potential games does not model
  wall-clock completion order or teammate-policy packet staleness.
- HAPPO/HARL supplies coordinated sequential heterogeneous-actor baselines;
  MAPPO remains a mandatory strong cooperative-MARL baseline.
- AFedPG is asynchronous training of one shared global policy, not distinct
  strategically interacting policy owners with self-fresh owner blocks.
- SMACv2 is preferred to legacy SMAC for the discrete layer because its
  procedural variation and partial observability require closed-loop policies.

The claim is not "the first asynchronous RL algorithm." It is the narrow
rate--coupling theory and empirical phase for distinct interacting Markov-game
policy blocks.

## 3. Benchmark layers

### Layer 0: local CPU integration smoke

Use PettingZoo MPE `simple_spread_v2` only to validate non-shared actor
ownership, packet snapshots, certified common-step application, transition
accounting, deterministic replay, and process teardown. It is not a headline
scientific result.

### Layer 1: continuous heterogeneous cooperative control

Use official HARL configurations as the starting point for MAMuJoCo
`Walker2d-v2`, `6x1`, and `Ant-v2`, `4x2`. A preflight may reduce this set only
for installation or deterministic-crash reasons recorded before any pilot
outcome; it may not select by observed method advantage.

### Layer 2: stochastic partially observable cooperation

Use SMACv2 `terran_5_vs_5` and `zerg_10_vs_10`. This tests recurrent policies,
stochastic unit composition, and partial observation as an empirical extension;
it is not covered by the tabular MPG theorem.

## 4. Algorithm family and baselines

Protocol-matched methods use non-shared actors and identical packet work:

1. fully utilized frozen-policy barrier;
2. raw single-flight async with the same base optimizer;
3. strong delay-adaptive asynchronous reference;
4. primary single-flight common-step learner (pathwise certified only on the
   finite-policy theorem class; empirical on the neural benchmark);
5. certificate ablation that removes cross-agent strategic coupling from the
   declared envelope.

External strong baselines retain their official synchronous semantics:

6. HAA2C;
7. HAPPO;
8. MAPPO without actor parameter sharing where the task permits.

The former sample-split strategic-debt/LSFF controller and oracle
directional-value controller are failed development branches, not core methods,
baselines, rescue mechanisms, or sources of outcome selection in this plan.

The theory-aligned implementation uses Monte Carlo actor returns with an
action-independent critic baseline frozen inside each packet. A PPO/HAPPO
variant is secondary and outside the finite-time theorem: clipping, Adam, and
a learned bootstrapped critic must not be silently claimed as covered.

Official HARL commit inspected for the design:

`b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`

The upstream checkout contains no root license file at that commit. Do not
vendor or redistribute HARL source; use a pinned external checkout and a small
auditable project-owned overlay while retaining upstream provenance.

## 5. Heterogeneous service and accounting

Artificial `sleep` is not sufficient main evidence. Report both endogenous
measured actor-packet time under declared CPU/GPU allocation and controlled
fixed per-actor workloads or CPU affinity that perform real, fully charged
rollout/optimization work. Every method receives the same actor-specific
packet workload.

Report logical event time, actual elapsed time, environment transitions,
optimizer FLOP proxy, cancelled partial work, barrier-cancelled work, and GPU
utilization. A shared-buffer official baseline may reuse a joint rollout; keep
that efficiency visible rather than hiding it as a protocol mismatch.

## 6. Metrics and preregistration ladder

Primary objects are actual time and charged work to fixed task-public return
thresholds; return versus actual time and transitions; final and lower-tail
return; policy-lag events and teammate KL drift at arrival; common-step scale
and certificate margin; CPU/GPU utilization; and actor idle fraction.

A GPU preregistration ladder may advance from smoke to pilot and then formal
confirmation only after the preceding stage is frozen and passes its declared
gates. The pilot may authorize formal seeds only if finite runs, exact work
identities, and no process leaks hold; the primary learner shows a reproducible
time-to-threshold advantage over the fully utilized barrier without a worse
charged-work tradeoff; its final return is noninferior to raw async; the
cross-agent-coupling ablation loses in registered high-drift cells; strong
HAPPO/MAPPO curves remain visible; and all subsequent seeds, thresholds,
analysis code, and source hashes are frozen before confirmation.

Exact numerical thresholds, seeds, task budgets, and Slurm layout belong only
to that later outcome-free preregistration. This document does not authorize a
GPU job.

## 7. Remaining pre-GPU work

The project-owned external-checkout runtime, Layer-0 CPU ownership/accounting
smoke, neural common-step interface and teammate-drift diagnostics now pass an
outcome-free, byte-exact local reproduction.  They remain empirical on the
neural actor; see `two_clocks_layer0_g0_validation.md` and
`two_clocks_neural_certificate_audit.md`.

The remaining gate is operational rather than a new CPU efficacy experiment:

1. perform outcome-free installation, task-shape, lifecycle and resource
   preflight on the selected HPC4 GPU stack;
2. freeze task budgets, genuinely new pilot seeds, numerical gates, runner and
   analyzer hashes, and Slurm layout in a separate immutable commit;
3. only then submit the minimal pilot to HPC4 scratch, with no `/project`
   output until validation passes.
