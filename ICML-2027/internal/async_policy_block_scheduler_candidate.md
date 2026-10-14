# Candidate redesign: interaction-aware asynchronous policy-block scheduling

Date: 2026-10-14
Status: research candidate only; no performance claim, experiment number, or
GPU authorization. This note does not alter or replace the frozen response
qualification or any TSP/V2/V3 artifact.

## Why the q-response route is not ready

The completed real-MAPPO qualification established a local critic-response
diagnostic, but its cross-stage gate failed. More fundamentally, its shadow
ranking observes q-specific candidate displacements after candidate training;
that is not an action-before-update signal. Recollecting q-specific probes
would require paying for data before choosing q and could erase the value of
selection. A fresh-sensor q study is therefore not the immediate next step.

## Candidate problem

Use CTDE with distinct actor parameters \(\theta=(\theta_1,\ldots,\theta_n)\).
All agents act from local observations at execution. A centralized trainer
receives asynchronous actor-gradient packets, each tagged with the policy
version under which its data were collected. A joint rollout batch naturally
contains data for every agent, so the scheduler can form all per-agent score
components without launching extra probe trajectories.

At a receipt event, the design variable is a subset of eligible policy blocks
\(S\subseteq[n]\) and their step sizes \(\alpha_i\). The candidate update is
\(u_i=\alpha_i\widehat g_i^{(v_i)}\) for \(i\in S\), and zero otherwise.
The interaction graph is a sparse representation of cross-block terms in the
local joint-return change, not an execution-time communication graph. In a
smooth local model,

\[
J(\theta+u)-J(\theta)
\ge \nabla J(\theta)^\top u-
\tfrac12\sum_i L_{ii}\lVert u_i\rVert^2-
\sum_{(i,j)\in E}L_{ij}\lVert u_i\rVert\lVert u_j\rVert-\epsilon_{\rm omitted}.
\]

The asynchronous penalty must explicitly cover version lag, sampling noise,
and omitted/dynamic graph interactions. This inequality is only a target
interface: no valid MARL-return bound has yet been proved for the implemented
actor/critic pipeline.

## Where Lyapunov drift would design the action

For message and environment budgets, introduce virtual queues
\(Q^m,Q^e\). At each packet-receipt event, minimize a valid upper bound on
\(\Delta\mathcal L_t - V\,\underline{\Delta J}_t\) over \((S,\alpha)\),
where \(\mathcal L_t\) includes queue quadratic terms and any justified
learning-state potential. Thus Lyapunov drift selects which actor blocks to
apply together and their step sizes; it is not merely a post-hoc convergence
analysis. The finite subset action is enumerable for small teams. Any claim
of sparse/large-team scalability requires a proved graph-approximation bound
and measured solver cost.

## What is and is not differentiated from prior work

- Xiao, Tan, and Amato, *Asynchronous Actor-Critic for Multi-Agent
  Reinforcement Learning*, NeurIPS 2022, already studies asynchronous MARL
  policies in decentralized, centralized, and CTDE settings. Async MARL or
  CTDE alone is not a novelty claim:
  https://proceedings.neurips.cc/paper_files/paper/2022/hash/1c153788756d35559c22d105d1182c30-Abstract-Conference.html
- Zhang, *Gradient-Aware Scheduling: Coupling Curriculum and Staleness for
  Async Reinforcement Learning*, ICML 2026, studies task difficulty and
  staleness allocation for asynchronous RL. Merely adapting staleness or
  learning rate is too close:
  https://proceedings.mlr.press/v306/zhang26bc.html
- Qin et al., *GradPS*, ICML 2025, uses gradient conflict to restructure
  parameter sharing. Gradient conflict alone is not sufficient novelty:
  https://proceedings.mlr.press/v267/qin25c.html
- Xie, Koyejo, and Gupta, *Zeno++*, ICML 2020, already scores candidate
  asynchronous updates by predicted loss descent. Generic candidate-update
  scoring is not sufficient novelty:
  https://proceedings.mlr.press/v119/xie20c.html

The only plausible differentiation is a theorem-backed, resource-constrained
choice of *which distinct strategic policy blocks may be applied jointly*
under stale actor packets, where cross-policy return interactions appear in
the certified drift and are available from the shared training batch. This
distinction is a hypothesis requiring a broader literature audit, not an
established novelty result.

## Mandatory CPU kill tests before any new GPU study

1. **Observable-path test:** from a single charged joint training batch,
   calculate every eligible actor gradient, stale version, and candidate
   subset/step score before applying any candidate update. The controller may
   not inspect candidate post-update parameters, held-out return, or future
   episodes.
2. **Independent ranking test:** use disjoint episode-seed clusters to test
   whether the pre-action score ranks joint subset updates by held-out return
   change. Report ties and confidence intervals; episode clusters, not
   timesteps or candidate subsets, are the independent units.
3. **Oracle-headroom test:** compare the held-out best subset with the
   strongest static schedule under identical message, total-transition,
   optimizer-row, and actor-update budgets. This is a ceiling diagnostic only.
   Freeze a practical-effect threshold and task set before outcomes. No return
   experiment proceeds if the ceiling is negligible or the score does not
   beat a no-skill / stale-age-only ordering out of sample.
4. **Certificate test:** establish simultaneous finite-sample coverage for
   stale-gradient bias and pairwise interaction remainders. If certification
   collapses to the empty update on ordinary contexts, stop or redesign.
5. **Compute test:** charge gradient/HVP/score and subset-optimization work;
   enumerate only where team size permits, and report scaling. No uncharged
   exhaustive oracle may be presented as the deployable controller.

Only after items 1--5 have a frozen protocol and CPU gates should a new
preregistration be considered. A positive synthetic result alone is not
standard-MARL evidence.
