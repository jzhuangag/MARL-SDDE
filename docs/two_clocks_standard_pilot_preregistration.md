# Two Clocks standard-environment pilot preregistration

## Status and scientific role

This document freezes the first outcome-bearing standard-environment test of
the **Two Clocks, One Game** mechanism.  It is a fresh-seed GPU pilot, not
formal evidence.  No outcome from this pilot may be used to change its tasks,
seeds, service profiles, methods, metrics, or mandatory gates.  Passing every
gate permits only the design of a separate formal preregistration; it does not
itself authorize a formal claim.

The experiment tests one coherent causal premise: under equal, fully charged
trajectory work, heterogeneous return times let single-flight asynchronous
actors query updated teammate policies more often than a fully utilized frozen
barrier.  The mechanism should therefore improve return as a function of a
controlled logical service horizon when strategic staleness matters.  Balanced
service is the phase control.  Actual Slurm duration is provenance only and is
not an outcome.

## Frozen implementation

Implementation base commit:
`c133025e2afef1926e5b48f3070fa293b788b7c5`.

| Artifact | SHA-256 of Git blob |
|---|---|
| `run_two_clocks_standard_pilot.py` | `ce1e575c728fa7da6efd531e2816a084271c41376c383529b39c98b6bc6c49fe` |
| `analyze_two_clocks_standard_pilot.py` | `d09fc189d3693d4950a78e3179fb666f82b9cf7651f1813be9e9438c67bfc92d` |
| `two_clocks_standard_pilot_config.json` | `b5ca6d7342adcf24008ec77d0e6a6af7ce0c23af2e769da6e9c7c465ea8f6775` |
| `test_two_clocks_standard_pilot.py` | `59a89f9127832a69b72e809710a6f9fa5f47b8257c1c1d4e32cfd61447927aa3` |
| `two_clocks_standard_pilot_a30.sbatch` | `3215007e13c38de23ea561e519651eb4210d1a49454f9fbeec831ea057e40aa7` |

External checkouts are HARL
`b1af98b0dbab72a2eee9d160751cd09aedbb8ce2` and SMACv2
`577ab5a2cff2391f8df582da5731ea9cd6adf3c6`.  The outcome-free HPC4 G0
gate passed at repository commit `af83e94f875f971cc5374a1e5f3f19658deec3f5`;
its final validation is recorded at commit
`9c9c8981218050868085cdfe23032483ddf04767`.

## Training and execution semantics

Each cooperative agent owns a distinct neural stochastic policy.  During
training, a centralized event coordinator launches at most one trajectory
packet per owner.  A returned packet is therefore fresh in its owner's policy
block but can be stale in teammate blocks.  Each packet contains a fixed
number of fully charged actor transitions.  If an episode ends inside a
packet, the environment is reset and collection continues; return-to-go is
reset at that episode boundary.  This prevents policy-dependent early
termination from changing the training work budget.

The training reward is cooperative.  There is no centralized critic and no
execution-time coordinator: after training, the distinct actors act from
their local observations.  Thus the experiment is centralized asynchronous
training with decentralized execution, not a claim about communication during
execution.

The neural actor uses the public HARL HAA2C architecture with hidden sizes
`[64,64]`.  The update is a frozen-baseline Monte Carlo policy-gradient step,
learning rate `5e-4`, clipped to Euclidean norm `0.02`.  This neural pilot is an
empirical extension.  It does not assert that an unconstrained neural network
satisfies the finite-policy Lyapunov certificate developed for the theorem
model.

## Frozen methods

1. `two_clocks_async`: apply each single-flight packet immediately on return.
2. `delay_scaled_async`: the same execution, with its step divided by
   `1 + 0.25 d`, where `d` is the observed sum of teammate policy-version
   increments since packet birth.
3. `frozen_barrier`: a strong fully utilized barrier.  Fast actors collect
   multiple independent packets during a round, all against the same birth
   joint policy; their steps are averaged once per owner at the barrier.

All three methods receive exactly the same deterministic, outcome-independent
packet opportunity set under a given task and service profile.  No sensing
trajectory, probe, or uncharged correction is available.

## Frozen tasks, clocks, and work

| Task | Agents | Block | Logical horizon | Balanced services | Heterogeneous services |
|---|---:|---:|---:|---|---|
| MAMuJoCo Ant-v2 4x2 | 4 | 25 transitions | 24 | `[1,1,1,1]` | `[1,1,2,4]` |
| SMACv2 Terran 5v5 | 5 | 30 transitions | 16 | `[1,1,1,1,1]` | `[1,1,2,2,4]` |

The exact per-row accounting is:

| Task/profile | Packets, every method | Async updates | Barrier updates | Charged environment steps including baseline |
|---|---:|---:|---:|---:|
| Ant/balanced | 96 | 96 | 96 | 2,500 |
| Ant/heterogeneous | 66 | 66 | 24 | 1,750 |
| SMACv2/balanced | 80 | 80 | 80 | 2,520 |
| SMACv2/heterogeneous | 52 | 52 | 20 | 1,680 |

The barrier is not idle: every actor keeps collecting packets.  Its lower
heterogeneous update count is adaptive query depth, not discarded data.  This
is the phenomenon under test and is why logical service time remains part of
the design even though physical job duration is not the primary metric.

## Frozen populations and seeds

There are four fresh pilot seeds per task, two profiles, and three methods:
24 rows per task and 48 primary rows total.  An isolated reproduction repeats
all 48 rows.  Seeds are:

- Ant: `91402731, 91402749, 91402763, 91402788`;
- SMACv2: `91402807, 91402829, 91402854, 91402873`.

The seed registry contains no formal seeds.  Initial policies, environment
seeds, stochastic actions, service opportunities, and evaluation schedule are
paired within task and seed.  Each row uses four fully charged baseline
blocks.  Evaluation occurs at logical fractions `0,.25,.5,.75,1` using two
episodes and is accounted separately from training.

## Frozen estimands

The primary outcome is normalized logical-time return AUC.  For candidate
`C` and baseline `B`, relative gain is

`(AUC(C) - AUC(B)) / max(abs(AUC(B)), 1)`.

Higher is better.  Terminal-return gain uses the same normalization.
Aggregates are unweighted means over the eight task-by-seed heterogeneous
cells.  With four pilot seeds per task, these are mechanism and feasibility
diagnostics; no confidence interval or significance claim is authorized.

## Mandatory gates

All gates are conjunctive.

1. **P1 validity/equal work:** 96 primary-plus-reproduction rows are finite;
   row population, hashes, seeds, packet counts, fixed transition charges,
   update counts, evaluation grid, and self-fresh invariant all match the
   frozen contract.
2. **P2 adaptive depth:** heterogeneous async/barrier update ratio is at least
   `1.5` in each task.
3. **P3 primary gain:** aggregate heterogeneous async-versus-barrier AUC gain
   is at least `3%`, and its taskwise mean is positive for both tasks.
4. **P4 directionality:** async beats the barrier in at least `5/8`
   heterogeneous task-by-seed cells.
5. **P5 async baseline:** aggregate async-versus-delay-scaled AUC gain is at
   least `-2%`, and async is strictly better in at least `4/8` heterogeneous
   cells.
6. **P6 phase control:** in each task, mean async-versus-barrier AUC gain under
   heterogeneous service strictly exceeds its balanced-service gain.
7. **P7 terminal safety:** taskwise heterogeneous terminal relative gain is
   no worse than `-10%`.
8. **P8 learning signal:** mean heterogeneous async terminal-minus-initial
   return is positive in each task.
9. **P9 clipping:** mean packet clipping fraction over all primary rows is at
   most `0.5`.
10. **P10 reproducibility:** primary and isolated reproduction JSON are
    byte-identical task by task.
11. **P11 provenance:** all four Slurm manifests pass `sha256sum -c`; source,
    external commits, runtime, GPU, command, job IDs, logs, and artifact paths
    are retained under the independent scratch root.
12. **P12 pilot/formal separation:** no formal seed or formal claim is present;
    every output has `formal_authorized=false`.

Any failed gate permanently stops formal escalation for this frozen pilot.
The failure may be studied, but no threshold, seed, task, method, profile, or
outcome may be changed and rerun under the same identifier.

## Compute and storage boundary

Four A30 jobs are planned: primary and isolated reproduction for each task.
Each job requests one A30, eight CPUs, 32 GB memory, and at most two hours.
The runner writes only small JSON, environment, log, and checksum files under
`/scratch/jzhuangag/MARL-SDDE-TwoClocks-20260902`.  It writes nothing to
`/home/jzhuangag` or `/project/vincentlau/jzhuangag`.

