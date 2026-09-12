# Exact timing contract for arrival-fresh MARL sensing

## Why an ordinary asynchronous rollout is not fresh

A policy-gradient measurement requires environment interaction.  If teammate
updates remain commit-enabled while an agent collects a nominally fresh
trajectory, the joint policy can change before that trajectory is applied.
The resulting estimator is another self-fresh/cross-stale packet, not an
arrival-fresh estimator.  Treating rollout generation as instantaneous would
invalidate the fusion theorem.

## Commit-barrier event

The executable centralized-training contract at completion event `k` is:

1. the server exposes current joint policy `theta_k` and the completed birth
   packet `b_k`;
2. LSFF computes its refresh decision from predictable information;
3. if no refresh is selected, the owner immediately applies `b_k`;
4. if refresh is selected, the server temporarily freezes parameter commits,
   snapshots `theta_k`, and fully charges independent on-policy trajectories
   generated from that snapshot;
5. it forms `f_k`, fuses `b_k` and `f_k`, applies the owner-block update while
   commits remain frozen, then releases the barrier.

Actors may finish other packets during the barrier, but those commits remain
queued.  Hence no policy block changes between the fresh snapshot and the
fused update, so `f_k` targets the theorem's `g_k`.  Decentralized execution is
unchanged and requires no such barrier.

## Resource and wall-clock accounting

Every barrier charges its actor transitions and serialized trainer time.  If
`s_k^F` is its duration, a pathwise wall-clock composition is

```
T_K <= T_K^async + sum_{k<K} u_k s_k^F.
```

The multi-resource queues price actor transitions and barrier duration
separately.  A hard remaining-transition check prevents finite-horizon budget
overshoot.  Pending completions retain their original actor finish times but
cannot commit before the barrier is released.

## Scope and novelty boundary

This contract turns freshness into an intermittent synchronization resource.
The work cannot claim novelty for synchronization, asynchronous actor-learners,
staleness-adaptive trust regions, validation gradients, or policy-divergence
triggers individually.  Its paper-level claim, if the remaining gates pass,
must concern their specific strategic-learning composition: distinct policy
blocks create self-fresh/cross-stale bias; a barrier buys an unbiased current
partial-gradient measurement; closed-form fusion quantifies its conditional
MSE value; and Lyapunov queues schedule that value under interaction and
wall-clock budgets with a potential/Nash finite-time consequence.

The contract is a necessary correctness condition, not yet a performance
result.  A barrier that is always selected reduces to synchronous training; a
barrier that is never selected reduces to raw asynchronous learning.  LSFF is
nontrivial only if it improves the return--resource frontier over both and over
strong fixed-period barriers.
