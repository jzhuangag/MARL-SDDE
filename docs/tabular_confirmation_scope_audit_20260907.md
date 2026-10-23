## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: theory--code claim-scope audit
- Origin Date: 2026-09-07
- Verification Status: SOURCE-VERIFIED SCOPE BOUNDARY
- Version Label: tabular_confirmation_scope_audit_v1

# Scope of the forecast-reversal confirmation

## Finding

PDSG-FR-001 is a valid confirmation of a delayed Markov factor-selection
mechanism, but it is not an end-to-end confirmation of the persistent
policy-cache Lyapunov state proposed for Pursuit.

In `certified_factor_async.py`, every event has a common owner update and a
candidate edge adds one state-dependent factor to that event's gradient.  The
joint controller sets

```text
reset_benefit_by_action = 0
receipt_cache_linear_upper = 0
outgoing_cache_weight = 0
```

for every candidate.  Thus the executed index contains predicted learning
alignment, packet-weight curvature, receipt-motion penalty, and the virtual
communication queue, but no persistent cache reset `B_p(a)` or cache energy
`H_p`.

The runner also coalesces a non-null action with zero packet weight into the
null action and charges no message.  This is correct for this abstract
one-event factor model: there is no cache state to change, so such an action
has no effect.  It is not the semantics of a real policy-cache refresh.

## Claims supported by the completed frozen confirmation

The frozen PDSG-FR-001 confirmation passed all twelve gates, including isolated
byte-exact reproduction. The list below is therefore an evidence statement,
subject to the stated finite-state scope, rather than a prospective claim.

The experiment supports all of the following within its finite-state model:

1. current-state myopic factor ranking can differ from the finite-horizon
   Markov ranking;
2. a sequential plug-in estimate can learn that forecast reversal from fully
   charged trajectory observations;
3. the same scalar Lyapunov index can jointly select a factor and delayed
   packet weight under a communication queue;
4. the favorable high-cycle phase can have broad equal-transition and
   communication-Pareto learning-risk gains over a frozen strong myopic
   comparator;
5. the low-cycle phase need not improve, so the result is not universal
   no-harm.

It does not by itself support:

1. a benefit from the persistent cache-energy term;
2. correct message accounting for a real refresh followed by zero update
   weight;
3. neural critic calibration or policy-induced trajectory kernels;
4. standard-MARL return improvement.

## Required Pursuit semantics

For a real worker cache, selecting `j -> i` copies the current actor `theta_j`
into `chi_(j|i)` before the rollout.  This transmission is charged by actor
payload bytes regardless of the eventual receipt-time owner weight.  The exact
cache reset

\[
 B_p(j\to i)=\frac{\beta_{ji,p}}2
 \|\theta_{j,p}-\chi_{j|i,p}\|^2
\]

must be applied at launch, and later owner/teammate updates must generate the
corresponding outgoing-cache increment.  If the resulting trajectory packet
receives zero weight, the refreshed cache still persists and can affect future
rollouts.

The Pursuit interface therefore needs separate invariant tests for:

- non-null refresh charging independent of `alpha`;
- exact before/after cache-energy change;
- persistence of the refreshed actor version;
- launch-time behavior generated from the selected cache profile;
- receipt-time application of the packet-carried weight;
- no double counting between cache motion and controlled-kernel error.

This scope distinction keeps the positive tabular mechanism result useful
without treating it as evidence for a state variable that the experiment did
not instantiate.
