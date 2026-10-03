# Multi-task participation-phase headroom preregistration

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan-to-run
- Origin Date: 2026-09-20
- Verification Status: FROZEN-STAGED-DESIGN
- Experiment: `TSP-MARL-MULTITASK-PHASE-A-001`

## Scientific question

The study asks whether two structurally different cooperative policy-learning
tasks contain a nontrivial finite-budget participation phase before a new
Lyapunov controller is trained.
It does not assume that either small or broad participation must win.
The outcome-aware endpoint oracle is used only as a development feasibility
diagnostic and is not a deployable comparator or manuscript claim.

## Outcome-free task selection

`simple_speaker_listener_v3` supplies discrete actions, heterogeneous roles,
explicit communication, and decentralized execution.
Parameter sharing is disabled because the speaker and listener have distinct
policy interfaces.

MaMuJoCo `HalfCheetah-v2/2x3` supplies continuous actions and heterogeneous
physical control under centralized training and decentralized execution.
Shared continuous exploration uses one public Gaussian innovation across
rollout workers while retaining each worker's Normal marginal.

The tasks were selected for interface and structural diversity, not from
observed returns under this experiment.

## Theory-derived resource rays

For rollout length `L`, server overhead `h`, and participation `q`, one update
costs `h+qL` messages and `L` parallel environment ticks.
The dimensionless boundary is therefore `B_msg/B_env = h/L+q`.
Both tasks use `h/L=4`, so the frozen ratios `5` and `12` are the exact endpoint
boundaries for `q=1` and `q=8`.
They define the message-binding and environment-binding rays without inspecting
return data.

## Stage-A lattice

The fixed endpoints are `q in {1,8}`.
The lattice contains two tasks, two resource rays, independent/shared rollout
coupling, and two new development seeds, for 32 runs.
All methods receive identical message and environment limits within a cell.
The horizontal axis is the maximum charged dual-budget fraction.

Independent workers use cyclically shifted environment seeds.
Shared workers use one environment seed and one public action innovation.
For discrete actions, inverse-CDF coupling preserves every categorical
marginal.
For continuous actions, standardized public Gaussian noise preserves every
Normal marginal while allowing worker-specific means and scales.

## Mandatory gates

All gates are conjunctive.

1. All 32 cells are unique, finite, exactly charged, and use the clean pinned
   HARL commit.
2. Each task has at least two distinct endpoint-oracle actions across its four
   coupling-by-ray cells.
3. For each task, the cellwise endpoint oracle improves return AUC by at least
   2 percent over that task's strongest single fixed endpoint.
4. On the message-binding ray, independent and shared coupling have different
   endpoint-oracle actions for each task.
5. Independent analysis replay is byte-identical.

Any failure stops full-q completion and controller training for this frozen
suite.
Passing authorizes only a separate outcome-free freeze for `q=2,4` and the
observable controller; it is not confirmatory evidence.

## Compute and storage

The experiment uses the pinned HARL environment already installed on HPC4.
Active outputs are written only under
`/scratch/jzhuangag/MARL-SDDE-TSP-MULTITASK-PHASE-A-001`.
No result is written to `/project` or `/home`.
