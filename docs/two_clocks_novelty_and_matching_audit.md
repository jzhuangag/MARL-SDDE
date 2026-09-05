# Two Clocks novelty and matching audit

Date: 2026-09-02.

Status: bounded primary-source confrontation and theorem-structure audit.  It
is not an exhaustive literature review or a final citation-integrity report.

## Reviewer-level novelty question

The proposed contribution is not "asynchronous SGD for MARL" and not a new
staleness-dependent learning rate.  Both would be too close to established
work.  The candidate survives only if the following package remains intact:

> a finite-time wall-clock phase theorem for distinct policy blocks in one
> Markov potential game, where a single-flight packet is self-fresh in its
> owner block but strategically stale in teammate blocks, with joint Markov
> packet bias/noise, an off-diagonal Lyapunov--Krasovskii upper bound, a
> cross-agent identifiability obstruction and a stochastic essential-agent
> clock lower bound.

Removing any of "distinct blocks", "joint Markov trajectory", "off-diagonal
self-freshness", or "wall-clock upper/lower phase" makes the current novelty
claim fail.

The Yu--Chen--Poor SDDE line makes a second boundary explicit.  Choosing the
number of active workers, group size, staleness threshold or gradient-dropout
policy from an SDDE characteristic root is already prior art for shared-model
ASGD.  Recasting that choice as a dynamic collaboration graph, or merely
replacing its Hessian norm by a Lyapunov weight, is not a defensible
contribution.  Here Lyapunov design is restricted to the off-diagonal
single-flight history and its common safe step; the paper-level object is the
distinct-policy rate--coupling phase.

## Nearest primary boundaries

| Work | What it already supplies | What remains different here | Risk |
|---|---|---|---:|
| Cohen et al., NeurIPS 2021, *Asynchronous Stochastic Optimization Robust to Arbitrary Delays* | Nonconvex delayed stochastic optimization with average-delay dependence and an efficient delay-robust rule | A shared stochastic objective/oracle; no distinct strategic policy blocks, Markov-game potential/Nash conversion, or owner-self-fresh decomposition | high |
| Mishchenko et al., NeurIPS 2022, *Asynchronous SGD Beats Minibatch SGD Under Arbitrary Delays* | Virtual-iterate analysis and asynchronous advantage over minibatch SGD under arbitrary delays | Workers query one shared model objective; the theorem does not separate diagonal owner freshness from off-diagonal teammate policy drift | very high |
| Adibi et al., AISTATS 2024, *Stochastic Approximation with Delayed Updates: Finite-Time Rates under Markovian Sampling* | Finite-time delayed SA under Markovian sampling and a delay-adaptive scheme | One contractive operator; no policy-induced joint Markov law, unilateral Nash criterion or wall-clock distinct-agent lower phase | very high |
| Yu, Chen and Poor, IEEE TSP 2025 (extending ICC 2024), *Distributed Stochastic Gradient Descent With Staleness* | SDDE/Poisson wall-clock analysis of shared-model ASGD; optimizes worker/group/dropout scheduling through learning rate, Hessian spectrum and staleness | Its workers update one global parameter and have no distinct policy ownership, owner self-freshness, off-diagonal strategic coupling, joint Markov-game trajectory or Nash/essential-policy-block phase | very high |
| Lan et al., ICLR 2025, AFedPG | Policy-gradient-specific asynchronous federated algorithm, convergence, sample and time complexity | Multiple workers update one global policy.  They are not policy owners whose updates rotate one another's gradients in a game | very high |
| Qu and Wierman, COLT 2020 | Finite-time asynchronous SA and Q-learning with coordinate coverage | Contractive SA/Q-learning rather than interacting policy-gradient blocks and a wall-clock barrier phase | medium |
| Maheshwari, Wu and Sastry, IEEE CSL 2024/2025 | Asynchronous decentralized actor--critic dynamics in general-sum/near-potential Markov games | Asymptotic decentralized learning; no finite-time packet-staleness, wall-clock barrier comparison or matching clock lower bound | high |
| Zhong et al., JMLR 2024, HARL/HAPPO/HATRPO | Sequential heterogeneous-agent updates and monotonic-improvement framework | Round/sequential update order rather than barrier-free, simultaneously in-flight policy packets | high |
| Xiao et al., NeurIPS 2022, *Asynchronous Actor-Critic for MARL* | MARL with asynchronous action durations | Execution-time action asynchrony rather than training-time optimizer packet completion | medium |

Official primary records used in this bounded audit are stored in
`two_clocks_novelty_sources.json`.  Several records already have independent
OpenAlex/Crossref/arXiv checks in the repository's earlier source audits.  A
fresh resolver and retraction pass is still mandatory before manuscript use.

## Strict separation from a black-box delayed-gradient certificate

Generic delayed optimization treats the full query point as stale.  A
single-flight distinct-policy packet has the structural identity

\[
 \theta_i^k=\theta_i^{b_k};
\]

only teammate blocks can have moved.  Consequently the Lyapunov history weight
uses the off-diagonal matrix `L_off`, while ordinary curvature still uses
`L_ii`.

This produces a strict family-level separation.  In a separable game
`L_off=0`, the self-fresh certificate permits

\[
 \alpha\le \min_i L_{ii}^{-1}
\]

for every packet event delay `D`.  A black-box proof that places diagonal
curvature inside the stale-history energy pays an artificial delay penalty and
shrinks its sufficient step as `D` grows.  The repository tests verify that the
self-fresh history is exactly zero in this case even at `D=100`.

This does not prove that no sophisticated coordinate-asynchronous theorem can
recover the same fact.  It establishes the technical object that a novelty
claim must defend: off-diagonal strategic staleness is not a relabelled scalar
delay.

## Upper/lower matching ledger

| Dependence | Upper side | Lower/separation side | Current status |
|---|---|---|---|
| stochastic accuracy | Markov packet variance/bias terms in the finite-time drift | Gaussian sign change-of-measure gives `sigma_i^2/Delta_i^2` packet requirement | matched at order on a subclass |
| essential-agent rate | deterministic coverage window or marked completion rate | `max_i packet_requirement_i/lambda_i` | matched qualitatively; constants/model differ |
| global barrier | fully utilized frozen-policy shadow batching | duplicate frozen queries cannot replace adaptive query depth | strict separation for a bounded family |
| cross-agent coupling | off-diagonal `L_ij` history weights and delay-dependent certified step | exact indistinguishable-game boundary `[s-B]_+^2/(2L)` | phase matched; delay/coupling rate not minimax-matched |
| Markov dependence | exact finite-horizon trajectory likelihood, truncation bias and packet bound | lower subclass is a one-state Gaussian oracle | upper only; richer Markov lower bound open |
| Nash criterion | conditional softmax occupancy/interiority conversion | stationarity-sign lower subclass | conditional; uniform Nash lower matching open |

The honest paper claim is therefore an instance-sensitive upper/lower **phase
picture**, not a globally minimax-optimal rate in every parameter.

## Decision

**Conditional novelty pass for one standard-benchmark attempt.**  The complete
package is plausibly ICML-level, but the risk of being judged a composition of
delayed SGD and MPG policy gradient remains material.

Before submission, at least one of the following must become strong in the
main text:

1. a sharper theorem showing that the off-diagonal single-flight bound is
   strictly better than the best applicable generic asynchronous result on a
   declared class; or
2. broad standard-MARL evidence that the predicted service--coupling phase is
   quantitatively useful and that the certified method preserves most of raw
   asynchronous speed without its high-coupling degradation.

Failure of both makes the work better suited to a theory/systems or control
venue than ICML.  No acceptance probability or "first" claim is authorized.
