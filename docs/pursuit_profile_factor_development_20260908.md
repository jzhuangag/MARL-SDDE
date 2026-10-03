## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: selected-packet critic development audit
- Origin Date: 2026-09-08
- Verification Status: SUMMARY-FEATURE CRITIC FAILED; RAW-OBSERVATION CRITIC ONLY REMAINING PURSUIT PATH
- Version Label: pursuit_profile_factor_development_v1

# Pursuit selected-packet profile-factor critic result

## Decision

The first deployable profile-factor critic fails three of four frozen
development gates.  It cannot authorize independent calibration, controller
efficacy, or GPU work.  Unlike the preceding privileged pair-head failure, its
training loss uses only actually selected, fully charged completed packets.

One positive diagnostic survives: the independently replicated test edge
effect is 5.79% of the mean absolute null-alignment scale.  Pursuit therefore
has nonzero local refresh headroom in this declared mismatch phase; the present
summary-feature critic does not identify it.

## Information separation

- Train seeds `94200--94215` supplied 1,024 selected packets.
- Validation seeds `94216--94219` supplied 256 selected packets and determined
  early stopping only.
- Test seeds `94220--94221` supplied 64 launches.  Eight copied-state
  counterfactual replicates were constructed only after the critic was frozen.
- The selected-packet conversion has an explicit taint test: replacing every
  stored counterfactual label by arbitrary values leaves all critic inputs and
  targets byte-identical.

The critic used a shared 35-dimensional launch-summary representation, a
48-unit self head, and a 48-unit profile-conditioned pair head.  It predicts
the cost return for the executed owner action.  Null and one-edge candidate
alignments are then evaluated by replacing only the selected donor profile in
one pair factor.

## Results

The best validation epoch was 69; training stopped after 120 epochs.  The
single-trajectory validation return `R^2` was `-0.987323`, already indicating
that the hand summaries did not form a useful conditional value state.

On untouched replicated conditional-mean test episodes:

| Gate | Threshold | Observed | Result |
|---|---:|---:|---|
| Edge-effect `R^2` | `> 0` | `-0.004386` | fail |
| Nonzero sign accuracy | `>= 0.60` | `0.424242` | fail |
| Best action accuracy, including null | `>= 0.55` | `0.451613` | fail |
| Mean edge effect / mean absolute null alignment | `>= 0.05` | `0.057887` | pass |

The mean absolute edge effect was `0.008667`; mean absolute null alignment was
`0.149722`.  There were 42 evaluated donor edges from 31
reference-available launches; 41.94% of those launches had a strictly positive
oracle refresh edge.

The ignored development artifact is
`tmp/policy_dependency_sync/pursuit_profile_factor_development_v1.json`,
SHA-256
`3BD4A4CB9AEECBADD932295BBC791685876CA6F1C4FF7127249CD08F322678F7`.
Its embedded source hash is
`44f5ace4396c7e651d3769d16d774d9db895ef4fe41d10dc26afc238ab495933`.
Wall time was `954.90 s` on four local CPU workers; no GPU or HPC4 resource was
used.

## What this changes

This failure cannot be rescued by more ridge values, a wider summary MLP, or
reuse of the test seeds.  The summary-feature critic is stopped.

The remaining scientifically distinct Pursuit estimator is a
parameter-shared raw-observation factor critic.  The current summaries discard
the spatial wall, pursuer, and evader configuration before the critic sees it,
whereas a standard centralized MARL critic normally receives the observation
tensor.  A small shared convolutional encoder can retain this information and
still score one-edge profile replacements in `O(Delta)` at fixed action count.
It must use new train, validation, and test seeds and the same four gate
thresholds.  This is the final estimator-class change permitted for Pursuit;
if it fails, Pursuit is removed as the principal efficacy benchmark rather
than repeatedly redesigned.

The Lyapunov theorem and the exact candidate algebra are unaffected: they are
estimator-agnostic and already carry a uniform critic/factor error term.  What
remains open is whether a standard observation-level critic makes that error
small enough to preserve a nontrivial action.
