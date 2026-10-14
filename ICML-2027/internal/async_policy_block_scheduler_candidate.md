# Candidate redesign: certified parallel blocks for sequential MARL updates

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

If an episode-cluster confidence argument supplies simultaneous errors
\(\lVert\widehat g_i-\nabla_iJ(\theta_t)\rVert\le\varepsilon_i\), then a
conservative action score would be

\[
B(S,\alpha)=\sum_{i\in S}\alpha_i
(\lVert\widehat g_i\rVert^2-\varepsilon_i\lVert\widehat g_i\rVert)
-\frac12\sum_{i,j\in S}\bar L_{ij}\alpha_i\alpha_j
\lVert\widehat g_i\rVert\lVert\widehat g_j\rVert,
\]

where \(\bar L\) is a positive-semidefinite majorant of the block-smoothness
matrix. The confidence radius must include sampling error and stale-version
bias; simply using the current minibatch gradient norm is not a certificate.
This score is computed before applying any packet. It does not itself prove
that a critic surrogate predicts policy return: that bridge must be supplied
by the exact policy-gradient estimator and the declared finite-horizon/Markov
assumptions.

For virtual-budget queues
\(Q^r_{t+1}=[Q^r_t+c^r(S_t)-b^r]^+\), \(r\in\{m,e\}\), the controller
would choose

\[
(S_t,\alpha_t)\in\arg\max_{S,\alpha\in\mathcal A_S}
\left\{V B(S,\alpha)-Q^m_t c^m(S)-Q^e_t c^e(S)\right\}.
\]

For each subset, the step-size subproblem is a concave QP when the majorant is
PSD; enumerating all subsets and solving each QP exactly is not a heuristic
catalogue scan, though it is exponential in team size. A large-team method
would need a separate approximation guarantee. This inequality and optimizer
are a proposed proof target only: no valid MARL-return bound has yet been
proved for the implemented actor/critic pipeline, and the simultaneous
confidence/mixing certificate is open.

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
- Zhong et al., *Heterogeneous-Agent Reinforcement Learning*, JMLR 2024,
  already has distinct heterogeneous policies and a sequential update scheme
  with monotonic-improvement theory (HAPPO/HATRPO/HAML). A scheduler over
  independent actor blocks is not enough:
  https://jmlr.org/papers/v25/23-0488.html

The candidate is therefore sharpened to a specific question: **can HAPPO's
provably safe sequential policy updates be batched and applied asynchronously
when a certified bound on cross-agent interaction permits it?** At each event,
the controller would choose a concurrently updated subset and step sizes,
while Lyapunov drift-plus-penalty trades certified joint-return improvement
against communication, environment work, and synchronization cost. The
cross-block interaction remainder is the proposed certificate that decides
which updates may safely commute. This is still only a novelty hypothesis:
the relationship to HAPPO's advantage factorization must be derived, not
assumed, and a broader priority/related-work review remains necessary.

One possible route to an actual return theorem is to treat a concurrently
updated block as a composite policy over its product action space, and apply a
HAPPO/HAML-style trust-region improvement step to that composite policy. If
blocks are then processed sequentially, the existing advantage-decomposition
logic may extend at the block level. This would provide a return-level
certificate without claiming that critic MSE is return. It does **not** yet
cover independently stale actor packets: those require valid joint
importance-ratio estimates or a proved staleness remainder, which may become
too conservative. Nor does simply grouping agents prove lower wall-clock time;
parallel hardware and fully charged synchronization/optimization measurements
are still required.

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
6. **HAPPO comparison:** use pinned HAPPO/HAML sequential updates as the
   theory-matched reference, not only MAPPO or fixed-color schedules. The new
   theorem must state when a parallel block update preserves monotonic
   improvement or quantifies its degradation relative to the sequential
   reference.
7. **Return-theorem route:** derive the block-level performance-difference
   inequality from the HAPPO/HAML advantage decomposition and joint
   trust-region penalty, then add stale-version correction. A critic-only
   Taylor expansion is not an acceptable substitute for this policy-return
   argument.

Only after items 1--5 have a frozen protocol and CPU gates should a new
preregistration be considered. A positive synthetic result alone is not
standard-MARL evidence.

## CPU implementation status

`experiments/async_policy_block_scheduler.py` implements the finite action
optimizer for the stated surrogate: exhaustive policy-subset enumeration and
a bounded concave quadratic step-size solve for each subset. Six tests cover a
closed-form one-block solution, interaction-aware subset selection, budget
queues, uncertainty-induced idling, PSD validation, and the exact outer
surrogate objective. The complete `ICML-2027/tests` suite passes (27 tests,
Python 3.11 / `ust2`). This validates only the optimizer implementation. The
gradient/mixing confidence certificate, policy-return bridge, MARL oracle
headroom, and practical scaling remain unimplemented and unverified.
