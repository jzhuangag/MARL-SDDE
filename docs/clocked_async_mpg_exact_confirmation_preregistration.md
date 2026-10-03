# Exact multi-state clocked-async MPG confirmation preregistration

Status: frozen design; no confirmation outcome existed when this document was
committed.

## Purpose

This is the first allowed CPU theory confirmation after the theorem gates.  It
asks whether the executable single-flight local-step rule exhibits the
predicted service-rate--coupling phase in a multi-state Markov potential game,
against a fully utilized frozen-policy shadow barrier.  It is not formal paper
evidence and does not authorize stochastic trajectory packets, a standard
MARL benchmark or GPU work.

## Frozen provenance

- config SHA-256:
  `2d2af65f81d50b488c93eb26f553412c8740446421d2612a71b09469007f5177`
- model source SHA-256:
  `6151964477847dc6786e3898ba48dde150fcea4ae70519afa159bbe51e25af70`
- runner source SHA-256:
  `517fd553d75169010587f783784312786ba88852ba316911b894685ed6c690fc`

The design commit must precede the first confirmation run.  No frozen file,
threshold, namespace or analysis rule may change afterward.

## Game and policies

The game has three states, two distinct binary-action softmax policy blocks,
an action-independent persistent Markov transition kernel and identical
discounted reward.  Local reward weights differ by agent; the second actor is
slower and begins nearer its preferred action.  Cross-agent reward coupling is
in `{0, 0.08, 0.16, 0.24}`.  Because the state dynamics are action-independent,
the exact discounted occupancy, potential, gradient and global block
cross-smoothness bounds are analytic.

Service ratios are `{1,2,4,8}`.  Durations are independently drawn from
`[0.9,1.1]` times each agent's public base duration.  The deterministic event-
delay bound is computed from these supports and checked on every run.

The theorem-facing learner has one in-flight packet per actor, applies the
exact block gradient on completion and uses the single-flight local-curvature
closed-form steps at 80% of their stability ceiling.  The comparator freezes
the joint policy each round, keeps every actor working until all have at least
one completion, counts all extra frozen-policy packets and applies one common
globally safe simultaneous update.

## Frozen population

- 16 cells: four couplings by four service ratios;
- primary cells: the 12 cells with service ratio in `{2,4,8}`;
- 64 fresh service seeds derived only from namespace
  `clocked-async-mpg-exact-confirmation-v1`;
- maximum wall-clock horizon: 120;
- target: normalized potential gap at most 0.15;
- total endpoints: `16*64*2=2048`.

The namespaces `development-tainted`, `development-tainted-2` and
`static-test` are excluded.  Their outcomes cannot enter confirmation metrics.

## Mandatory pre-reproduction gates

1. `C1`: every endpoint is finite and every realized delay is within its
   deterministic registered bound.
2. `C2`: paired target-time coverage is at least 95% of endpoints.
3. `C3`: across the 12 primary cell medians, the geometric async/shadow time
   ratio is at most 0.80.
4. `C4`: async is strictly faster in at least 10 of 12 primary cells.
5. `C5`: within every coupling, the ratio at service ratio 8 is below that at
   service ratio 2.
6. `C6`: within every primary service ratio, coupling 0.24 has a ratio no
   smaller than coupling 0, exposing the predicted coupling cost.
7. `C7`: the geometric primary final normalized-gap ratio is at most one.
8. `C8`: every completed shadow packet and every async packet is counted;
   packets cannot be less than applied updates.
9. `C9`: confirmation and development namespaces are disjoint.
10. `C10`: a clean rerun in a new output directory must reproduce endpoints
    and summary byte-for-byte.

Any failure stops before stochastic Markov packets, formal seeds or GPU work.
Thresholds will not be rounded or relaxed.

## Commands

```text
.venv/Scripts/python.exe -m experiments.clocked_async_mpg.run_exact_multistate_confirmation validate
.venv/Scripts/python.exe -m experiments.clocked_async_mpg.run_exact_multistate_confirmation run --output-dir <new-dir>
```

Results directories are ignored artifacts.  Only the post-run validation,
hashes and an honest stop/continue decision may be committed after execution.
