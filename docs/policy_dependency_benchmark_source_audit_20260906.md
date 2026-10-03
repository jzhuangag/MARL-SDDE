# Standard-MARL policy-dependency source audit

Date: 2026-09-06

Status: outcome-free source and environment feasibility audit.  No policy was
trained and no method return was observed.

## Decision

Use six-agent HalfCheetah with the standard Multi-Agent MuJoCo `6x1` actuator
partition as the first CPU dependency-tail development task.  Its declared
candidate graph is the published joint-chain graph, whose maximum block degree
is two.  The task remains only an approximate-sparsity candidate: global rigid-
body dynamics and shared return mean that the source graph does not prove a
zero true policy-gradient tail.

Keep cooperative RWARE and shared-reward LBF as dense controls.  Do not use
MAgent2 as a positive task under the current theorem, because its battle and
combined-arms environments are competitive and do not share the cooperative
potential objective assumed by the finite-time result.  Do not claim that a
local observation window by itself implies a sparse policy-gradient graph.

## Immutable source inventory

| Source | Commit inspected | Relevant fact |
|---|---|---|
| [Multi-Agent MuJoCo](https://github.com/schroederdewitt/multiagent_mujoco) | `b212ddd74b258e7cea006ff1d642b5ffada4b99d` | HalfCheetah has a six-joint chain and a registered `6x1` partition; the environment repeats one shared MuJoCo reward to all agents. |
| [Level-Based Foraging](https://github.com/semitable/lb-foraging) | `79f383b7db0a3530f14252d5b6cc0d85aa9fb385` | Default rewards are assigned to participating loaders; a shared-reward cooperative variant is available, but sharing introduces global return coupling. |
| [Multi-Robot Warehouse](https://github.com/uoe-agents/robotic-warehouse) | `43018983b5e42cd8050481a871eca0baf556ac7e` | Observations use a finite sensor range, while `RewardType.GLOBAL` broadcasts each delivery reward and the request queue is shared. |
| [MAgent2](https://github.com/Farama-Foundation/MAgent2) | `0d2e0e344fa84411eeba4baf03dc3b7273c4f14d` | Battle-style tasks have local view/attack ranges and local event rewards but are competitive many-agent games. |
| [HARL](https://github.com/PKU-MARL/HARL) | `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2` | Its PettingZoo MPE wrapper sums all per-agent rewards and repeats the total to every agent; `simple_spread_v2` is therefore a dense control for the present claim. |

The source repositories were cloned at the listed revisions only for this
audit under `tmp/benchmark_source_audit_20260906`; they are ignored provenance,
not vendored project dependencies.

## Candidate 1: MAMuJoCo HalfCheetah `6x1`

The original benchmark partitions the six HalfCheetah actuators into

\[
(bfoot),(bshin),(bthigh),(ffoot),(fshin),(fthigh)
\tag{1}
\]

and declares the chain

\[
bfoot-bshin-bthigh-fthigh-fshin-ffoot.
\tag{2}
\]

At the policy-block level, the prospective support therefore has maximum
degree two.  This gives a non-arbitrary sparse candidate graph before any
learning result is seen.  The task is cooperative and naturally uses distinct
policy blocks whose actions jointly control one body, matching centralized
training with decentralized execution.

However, Equation (2) is a kinematic/observation graph, not a theorem about the
true policy gradient.  MuJoCo's mass matrix and the velocity-based shared reward
can transmit an action's effect beyond one adjacent joint.  The next audit must
measure the omitted cross-policy influence and compare the physical chain with
degree-matched random graphs.  If the measured tail is not compressible, this
task becomes a dense control rather than a positive result.

Local execution feasibility is already established without installing a new
package: conda environment `ust2` contains Gymnasium 1.0.0 and MuJoCo 3.3.0,
and `HalfCheetah-v5` reset plus one transition completed on CPU with observation
dimension 17 and action dimension 6.

## Why the other candidates are not promoted

### Cooperative RWARE

The finite sensor window is attractive for scalable actors, but the global
reward mode adds a unit delivery reward to every agent and the request queue is
shared.  Thus distant policies can affect an owner's return even when they are
outside its current sensor window.  Individual reward avoids that immediate
density but changes the paper to a general-sum objective.

### Level-Based Foraging

The default individual reward is local to agents that jointly load a food,
which gives meaningful dynamic interaction neighborhoods.  It is nevertheless
mixed cooperative--competitive.  The shared-reward variant restores a common
objective but makes all remote collections enter every agent's return.  LBF is
therefore useful as a theorem-boundary control, not the first positive task.

### MAgent2

MAgent2 provides the strongest large-scale locality: battle agents have finite
view and attack ranges.  Its standard tasks are competitive, so replacing the
potential loss by a general-sum gradient mapping would require a different
monotonicity/variational-inequality theorem.  That extension is scientifically
interesting but would broaden the current project before the cooperative core
is validated.

### HARL PettingZoo MPE

The inspected HARL wrapper explicitly sums all agent rewards and repeats the
total.  Earlier project experiments already used three-agent
`simple_spread_v2`; with only three blocks and a global reward it cannot
substantiate the new sparse-complexity claim.  It remains a dense small-agent
sanity control.

## Next admissible CPU audit

The next development run will not train a controller.  Under fixed random
smooth policies and common random numbers, it will estimate how the owner
finite-horizon policy-gradient direction changes when one teammate policy
version is perturbed.  It will compare:

1. the MAMuJoCo physical chain;
2. degree-matched random supports;
3. the complete graph;
4. horizon and perturbation scales;
5. finite-difference resolution and seed uncertainty.

The decisive quantity is the fraction of total cross-policy influence retained
by the physical neighbor set, together with a nonzero best-edge margin.  This
is a design-stage dependency-tail measurement, not a return comparison.  A
separate preregistration is required before execution.
