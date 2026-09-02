# Handoff: outcome-free standard-MARL gate for Two Clocks

Date: 2026-09-02.

Status: execution handoff, not a scientific preregistration and not permission
to run outcome-bearing seeds immediately.  The receiving agent must complete
Stage G0 and create a separate immutable preregistration commit before any
scientific pilot.  Formal seeds are not authorized.

## 1. Fixed scientific scope

The surviving question is:

> When heterogeneous learners own distinct interacting policy blocks, can a
> barrier-free learner exploit self-fresh updates while controlling the
> strategic staleness caused by teammate-policy motion?

The protocol is centralized event-driven training with decentralized execution
(CTDE).  Agent `i` alone writes actor block `theta_i` and has at most one
fixed-work packet in flight.  When its packet returns, its owner block is still
self-fresh; only teammate blocks may have changed.  The packet is applied
immediately and the owner launches its next packet from the current joint
policy.

The finite/tabular theorem chooses one common step from a
Lyapunov--Krasovskii interaction envelope.  The unconstrained HARL neural actor
does not satisfy the current uniform neural certificate.  On MAMuJoCo and
SMACv2 the timing and common-step mechanism is therefore an explicitly
empirical extension.  Do not call it theorem-covered neural MARL.

Do not revive any of the following as the main algorithm: participation-`q`
selection, dynamic collaboration graphs, strategic-debt/LSFF control,
value-of-information sensing, paid perfect sensing, noisy secant control, or
the failed EXP-017/T-083 branches.

## 2. Exact source boundary

1. Use the GitHub repository `jzhuangag/MARL-SDDE`, branch
   `codex/joint-ms-exp007c`.
2. Checkout the exact remote commit reported with this handoff and record
   `git rev-parse HEAD`, `git status --short`, the patch hash, and every command.
3. Use the pinned external HARL checkout
   `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`.
4. Do not vendor or redistribute HARL.  Keep a small auditable overlay in this
   repository and record upstream provenance.
5. Read before acting:
   - `docs/two_clocks_publication_theorem_chain.md`;
   - `docs/two_clocks_neural_certificate_audit.md`;
   - `docs/clocked_async_mpg_standard_marl_bridge.md`;
   - `docs/two_clocks_novelty_and_matching_audit.md`;
   - `docs/two_clocks_one_game_status.json`.

Use the `$hpc4` skill before every HPC4 read, transfer, run, monitor, or cleanup
operation.  Use `/scratch/jzhuangag/MARL-SDDE` for active work.  Stage G0 and
the pilot must not write results to `/project`; durable archival requires a
later explicit decision after validation.  Check `/scratch`, `/home`, and
`/project` capacity before staging.  Preserve all existing artifacts.

## 3. Stage G0: outcome-free integration and resource preflight

G0 may execute only shape, lifecycle, accounting, deterministic-replay and
resource-measurement checks.  It must not compare task return, win rate,
time-to-return, learning curves, or method advantage.

Implement and test the following invariants:

1. every non-shared actor block has exactly one writer;
2. at most one packet per actor is in flight;
3. a packet stores its birth joint-policy version and owner version;
4. the owner version at arrival equals its birth owner version;
5. completion metadata is not computed from trajectory reward or gradient;
6. packet work is fixed before launch and is identical across
   protocol-matched methods;
7. completed, cancelled and barrier-discarded transitions are all charged;
8. wall time uses a monotonic clock around the complete learner process;
9. worker teardown leaves no live process, CUDA context or orphan shared
   memory;
10. fixed-seed smoke replay gives identical packet/event/accounting traces;
11. one neural update exposes arrival lag, teammate-policy KL, applied step,
   optimizer work proxy and certificate/proxy margin without claiming a
   uniform theorem;
12. external baselines retain their native synchronous semantics and their
   shared-rollout efficiency remains visible.

Use PettingZoo MPE `simple_spread_v2` for local CPU integration only.  For GPU
resource preflight, target at least MAMuJoCo `Ant-v2`, `4x2`, and SMACv2
`terran_5_vs_5`.  Also attempt the planned `Walker2d-v2`, `6x1`, and
`zerg_10_vs_10`.  A task may be removed only for an installation failure or a
deterministic crash documented before outcome access; never select tasks by
observed advantage.

Measure actual packet service distributions, GPU memory, CPU memory, disk,
utilization and teardown time.  Use endogenous task/model work and, if needed,
controlled actor-specific real workloads or CPU affinity.  Artificial `sleep`
is not acceptable main evidence.

G0 must end with:

- an environment lock/spec and CUDA/driver/Slurm inventory;
- source and overlay hashes;
- test and smoke logs;
- a task-by-task installation decision made without outcome comparison;
- measured resource estimates for the smallest pilot;
- confirmation that no scientific endpoint or return comparison was produced.

If any ownership, accounting, lifecycle or deterministic-replay invariant
fails, stop before preregistration.

## 4. Separate immutable pilot preregistration

Only after G0 passes, create and push a standalone preregistration commit.  It
must freeze, before outcome access:

- exact repository, overlay, runner, analyzer and configuration hashes;
- retained tasks and the outcome-free reason for any removal;
- packet work, optimizer, network, critic, rollout horizon and all budgets;
- actual-time measurement and charged-work identities;
- completely new pilot seeds isolated from all development/formal seeds;
- Slurm partition, GPU/CPU/memory/time request and array layout;
- exact policies, comparisons, aggregation rules and missing-run handling;
- task-public return thresholds or an outcome-free rule deriving them;
- numerical gates and the rule that any mandatory failure stops formal;
- artifact layout, SHA-256 manifest and byte-level analyzer replay.

Do not alter gates, tasks, budgets, seeds, missing-run rules or analyzers after
the first scientific outcome exists.

## 5. Frozen method family for the pilot

Protocol-matched methods must use non-shared actors and identical declared
packet work:

1. fully utilized frozen-policy barrier;
2. raw single-flight asynchronous updates with the same base optimizer;
3. a strong delay-adaptive asynchronous reference;
4. Two Clocks single-flight common-step neural extension;
5. a coupling-envelope ablation that removes off-diagonal teammate motion.

Also report official HAA2C, HAPPO and non-parameter-sharing MAPPO where the task
supports them.  Do not disguise native shared-rollout efficiency as unfair.
Do not tune the primary method or its ablation on the pilot outcomes.

The primary neural method may use a declared measurable teammate-KL or local
Jacobian diagnostic, but it must be labelled a practical proxy.  It may not be
presented as the finite/tabular pathwise certificate.  A theorem-covered neural
variant would require a separately frozen projected/spectral architecture,
bounded Gaussian variance, diagonal smoothness and clipping-bias analysis.

## 6. Required measurements and nontriviality

Report for every method and task:

- return/win rate versus actual elapsed time and charged transitions;
- time to each preregistered public threshold;
- final return, lower-tail return and instability/divergence rate;
- total, completed, cancelled and barrier-discarded transitions;
- packet service time, arrival lag and teammate-policy KL at arrival;
- applied common-step scale and practical margin/proxy;
- actor idle fraction, CPU/GPU utilization and controller overhead;
- seed-level endpoints and complete trajectories.

Passing only a no-harm ratio is not sufficient: a method that always emulates
the baseline passes trivially.  The preregistration must include a positive
nontriviality gate.  At minimum, require reproducible actual-time improvement
over the fully utilized barrier on both continuous-control and stochastic
partially observable task families, no material charged-work regression,
final-return noninferiority to raw async, a registered loss for the
no-coupling ablation in high-teammate-drift cells, and correct phase directions
when service heterogeneity or interaction strength changes.

Use exact numerical tolerances justified by G0 measurement precision and
public task scales, never by scientific pilot outcomes.  Raw async is allowed
to be faster than the conservative common-step method; the paper must show the
full stability--speed Pareto frontier rather than hide that comparison.

## 7. Execution and stopping policy

Run the smallest A30 pilot capable of exercising both task families.  Pilot is
not formal evidence.  On completion:

1. verify Slurm exit codes and inspect logs for NaN, OOM, process leaks and
   accounting errors;
2. copy nothing to `/project` until validation passes and archival is
   separately authorized;
3. generate SHA-256 manifests for runner, configs, endpoints, trajectories,
   summaries and logs;
4. rerun the frozen analyzer from raw artifacts and require identical output;
5. publish all gates, including failures, without rounding across thresholds;
6. stop formal if any mandatory gate fails; do not rescue by changing a task,
   seed, budget, comparator, threshold or experiment identifier.

Return the exact commit, job IDs, environment, commands, logs, artifact paths,
checksums, per-gate table, and a direct scientific decision.  Formal execution
requires a subsequent explicit authorization.

## 8. Copyable prompt for the HPC4-connected agent

```text
Use the $hpc4 skill and work autonomously on the Two Clocks standard-MARL
bridge in jzhuangag/MARL-SDDE. Pull branch codex/joint-ms-exp007c and checkout
the exact remote commit supplied by the sending agent. Read
docs/HANDOFF_TWO_CLOCKS_GPU.md completely and obey its stage separation.

First perform only Stage G0: build a small external-checkout overlay against
HARL commit b1af98b0dbab72a2eee9d160751cd09aedbb8ce2, validate single-writer and
single-flight ownership, packet-version identities, transition/cancelled-work
accounting, deterministic replay, process teardown, and measure resources on
the specified local smoke and GPU task shapes. Do not compare scientific
returns or method advantage in G0. Use /scratch/jzhuangag/MARL-SDDE; do not
write pilot results to /project and do not delete or modify existing artifacts.

If and only if every G0 invariant passes, create a separate immutable,
outcome-free pilot preregistration commit freezing new seeds, tasks, budgets,
source/config/analyzer hashes, methods, exact numerical gates, Slurm layout,
artifact checksums and the stop rule. Push that commit and report it before
submitting scientific pilot jobs. Do not run formal seeds. Do not claim that
the unconstrained neural actor is covered by the finite/tabular Lyapunov
theorem. Preserve all negative results and stop immediately on any mandatory
failure. Return commands, commit/diff, job IDs, logs, artifacts and SHA-256
provenance.
```
