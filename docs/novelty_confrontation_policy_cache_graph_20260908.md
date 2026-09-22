## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: primary-source novelty confrontation
- Origin Date: 2026-09-08
- Verification Status: BOUNDED NOVELTY HYPOTHESIS SURVIVES; EMPIRICAL/THEORETICAL PACKAGE OPEN

# Novelty confrontation for Lyapunov-scheduled policy-cache graphs

## Decision

The mainline is not subsumed by the primary works found in the targeted search,
but none of its ingredients is individually new. The defensible unit of
novelty is the **training-time causal control problem**:

> In asynchronous CTDE with distinct interacting policies, choose which
> recipient-specific teammate policy version a rollout owner receives and how
> strongly the resulting delayed owner packet is applied, using one learning
> Lyapunov drift and an explicit policy-byte debt queue.

This survives as a bounded novelty hypothesis only. If the learned controller
cannot recover a material part of the independently confirmed dynamic-cache
value on standard tasks, the distinction is architectural but not an ICML-level
contribution.

## What prior work already owns

- SchedNet learns which agents broadcast execution-time observation messages
  under a shared medium. Bayesian ego-graphs and dynamic coordination graphs
  likewise learn execution-time communication/value-factorization graphs.
  Therefore “learning a dynamic MARL graph” is not our novelty.
- FACMAC uses a factored centralized critic and deterministic centralized
  policy gradients. Therefore local factorization and continuous policy
  gradients are inherited machinery.
- Lin et al. prove scalable actor--critic learning when the *environmental
  state/reward dependency graph* is stochastic and nonlocal. Therefore a
  stochastic interaction graph or local `Q`-decay is not our novelty.
- ACAC treats asynchronous macro-action decision times in CTDE. Min et al.
  study asynchronous communication among homogeneous learners of a common
  linear MDP. Therefore “asynchronous MARL” is not our novelty.
- Recent single-policy asynchronous RL work already controls stale-gradient
  alignment, derives staleness--learning-rate scaling, or adapts a trust
  region. Therefore gradient alignment and stale-packet weighting alone are
  not our novelty.

## Exact separation

The proposed graph is neither the physical/environmental dependency graph nor
the execution-time message graph. For every directed pair `j -> i`, it records
which **version of actor `j`** owner `i` will use in its next training rollout.
Refreshing that edge changes the controlled trajectory law and hence the
gradient packet returned for owner `i`; it does not alter decentralized actor
execution after training.

The online action is `(edge, alpha)`. The edge changes the behavior-policy
profile; `alpha` changes the receipt-time update. The same drift upper bound
chooses both, while a virtual queue prices actual policy bytes at launch. The
statistical target is revealed only for the selected trajectory on receipt.
This launch/cache/kernel/packet chain is absent from the compared methods.

## Reviewer-facing kill conditions

The distinction is too narrow for ICML if the paper contains only a cache
state machine, an oracle, or a generic drift inequality. A credible package
requires all of the following:

1. a learned causal controller, not future-return or counterfactual labels;
2. material equal-resource improvement over complete, no-refresh, age,
   mismatch, periodic/random, strong fixed, and strong online envelopes;
3. at least two standard tasks plus a dense/locality degeneration and scaling
   in delay, degree, agent count, and communication budget;
4. a finite-time convergence/stationarity theorem whose approximation,
   Markov, delayed-feedback, owner-motion, and topology terms correspond to
   logged quantities;
5. evidence that joint edge/weight control beats edge-only and weight-only
   ablations, rather than merely restating either prior line;
6. a fresh submission-time systematic search, since 2026 asynchronous-RL work
   is evolving quickly.

The current finite-state confirmation and Multiwalker return-oracle
confirmation meet neither condition 1 nor the entire condition 2. The running
alignment gate asks whether the theorem-facing signal has enough intrinsic
headroom to justify building condition 1.

## Search boundary

Targeted queries covered asynchronous MARL, policy staleness, policy-version
caches, dynamic/learned communication graphs, factored centralized policy
gradients, and Lyapunov scheduling. Search results and bibliographic metadata
were checked against official PMLR, NeurIPS, OpenReview, or arXiv records in
`citation_verification_policy_cache_graph_20260908.json`. No bibliography entry
was added or edited. This is a confrontation audit, not a claim of exhaustive
priority.
