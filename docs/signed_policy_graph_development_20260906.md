## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-09-06
- Verification Status: ANALYZED
- Version Label: signed_policy_graph_development_v1

# Signed policy-version graph: actual-learning development decision

## Decision

Retain one candidate only:

> fixed-horizon, fixed-cap asynchronous learning with a launch-time signed
> policy-version graph chosen by paired Lyapunov drift-plus-penalty.

Stop online horizon selection and unsigned packet-debt graph selection as the
primary algorithm.
Their certified-debt headroom did not translate into accumulated learning
risk.
The surviving signed rule clears the internal exact-model terminal-risk
headroom target against a substantially strengthened comparator envelope, but
it still uses the true quadratic drift and is therefore an oracle development
ceiling rather than an executable MARL result.

## Model and accounting

The event-driven cooperative quadratic Markov game has six distinct policy
blocks, persistent teammate-version caches, a moving donor dependency,
random packet completion delays, two Markov/noise regimes, and owner-sharded
asynchronous receipts.
The fixed horizon is `H=4` and every fixed-cap policy is charged exactly four
actor transitions per launch.
No extra receipt-control trajectory is charged because the surviving rule
does not use one.

Every replicate applies a common random permutation to the six initial policy
coordinates and target coordinates.
This preserves the quadratic spectrum and difficulty while preventing an
arbitrary agent numbering from making one fixed relative offset uniformly
privileged.
Common random numbers are shared across policies within a seed.

The signed action is null or one causal-cone edge, so its exact oracle scan is
`O(Delta)`.
Its learning weight is `sqrt(number of launches)`.
The message queue is updated at every launch.

The strong comparator family contains no refresh; unsigned packet debt;
largest mismatch; current active donor; largest conditional coefficient;
top-two mismatch and coefficient; round-robin; periodic full refresh; every
fixed one-offset graph; and every fixed two-offset graph.
Non-null fixed policies use the same token-bucket message budget, allowing a
two-edge graph to spend accumulated budget rather than being discarded when
its instantaneous cost exceeds the per-launch average.
The reported envelope selects the lowest-risk feasible comparator separately
in each cell, which favors the baseline.

## Rejected branch

The joint online horizon/unsigned-debt rule first showed 25.13% median
certificate-debt headroom in an outcome-free static scan.
When actual parameter updates and random receipts were added, its cumulative
risk was worse than the then-current strong comparator by 2.20% at 80 launches
and 6.30% at 240 launches.
A full-cap variant with online horizon and graph was also worse in cumulative
risk by 1.80% and 1.47%, respectively.
The cause is structural: minimizing birth-time unsigned debt does not price
the signed optimization effect or the number and timing of usable updates.
The static debt result is preserved as a certificate audit and is not promoted
to learning evidence.

## Signed oracle development results

The principal binding-budget grid contains 24 cells: coupling in
`{0,0.6,0.9}`, maximum extra delay in `{2,5}`, message budget in `{0.5,1.0}`,
and regime-switch probability in `{0.03,0.12}`.
There are 16 active and eight uncoupled control cells, four development seeds,
and 80 launches per trajectory.

| Metric | Observation |
|---|---:|
| active terminal-risk median improvement | 10.3406% |
| active terminal-risk strict improvements | 16/16 |
| active terminal-risk minimum improvement | 5.2994% |
| active cumulative-risk AUC median improvement | 1.4414% |
| active cumulative-risk strict improvements | 16/16 |
| maximum absolute uncoupled terminal difference | 0 |
| maximum absolute uncoupled AUC difference | 0 |
| median messages/launch at budget 0.5 | 0.4125 |
| median messages/launch at budget 1.0 | 0.740625 |
| maximum message-budget excess | -0.059375 |
| maximum environment-budget excess | 0 |
| minimum mean active graph supports | 10.75 |

Terminal improvement is positive in every registered delay, coupling,
message-budget, and switch-speed group.
Its median is 9.3349% at budget 0.5 and 12.3186% at budget 1.0; 6.6647% at
coupling 0.6 and 15.4584% at coupling 0.9.
Thus the active aggregate clears 10%, while the weaker-coupling and tighter-
budget strata are correctly harder.

The earlier nonbinding 32-cell grid gave 12.3186% median terminal improvement,
32/32 strict terminal improvements, 1.7494% median cumulative-risk
improvement, and 32/32 strict cumulative-risk improvements after all fixed
one- and two-edge comparators were included.

Ignored raw outputs:

- `tmp/policy_dependency_sync/async_packet_game_signed_ratebaselines_80s4.json`,
  SHA-256 `24482D363994CDB0A21CA852C246771DFD6D49933972EFE6E719DAA02A39175E`;
- `tmp/policy_dependency_sync/async_packet_game_budget_80s4.json`, SHA-256
  `77F64A6D9D81F7974F257194239D9958C29A109663D42CD53AD2A83E12DC4DFA`.

## Theory alignment

The companion paired theorem defines the launch score as the expected
fixed-step smoothness drift plus an explicit launch-to-receipt gradient-motion
bound.
The virtual queue and signed packet score are minimized in one DPP index.
After a terminal drain, launch/receipt pairs telescope even when receipts are
out of order.
The theorem provides a finite-time stationarity bound relative to a
budget-feasible causal comparator, with explicit score-estimation and delay
remainders.

This removes three earlier mismatches: no online horizon is claimed from an
unsigned surrogate, no noisy receipt root is required, and the graph action is
linear rather than exponential in local degree.

## Statistical and methodological fallacy scan

Coverage: 11/11.

1. Development seeds are not described as independent formal evidence.
2. The 10% terminal effect is not substituted for the smaller AUC effect.
3. All 16 active and all eight uncoupled cells are retained.
4. Common random numbers are not counted as additional samples.
5. The baseline is selected outcome-wise per cell, which is conservative for
   the proposed rule and is stated explicitly.
6. Agent relabeling removes an arbitrary fixed-offset confound and is shared
   by every policy.
7. The stopped unsigned branch remains recorded rather than hidden.
8. Exact quadratic drift is labelled oracle information, not an observable
   neural critic.
9. Message, environment, terminal-risk, and AUC claims are reported
   separately.
10. Uncoupled equality is not generalized to all stationary or dense games.
11. No p-value, confidence interval, or formal acceptance claim is made from
    four design seeds.

## Next gate

The next CPU stage is a separately frozen exact-oracle confirmation with new
seeds, a binding budget, the same fixed horizon, null-or-one-edge action, full
strong comparator envelope, and byte-exact reproduction.
It must preserve terminal and cumulative metrics and all controls.
Passing it authorizes integration of the already validated expectation-level
sparse signed estimator into the asynchronous learner; it does not by itself
authorize a paper claim or GPU result.

No GPU or HPC4 work is required for this gate.
