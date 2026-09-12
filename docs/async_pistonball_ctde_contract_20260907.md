## Material Passport

- Origin Skills: academic-research-suite; publishable-academic-writing
- Origin Mode: implementation and theory alignment
- Origin Date: 2026-09-07
- Verification Status: VERIFIED DEVELOPMENT CONTRACT; NO EFFICACY CLAIM
- Version Label: async_pistonball_ctde_contract_v1

# Asynchronous Pistonball CTDE contract

## Decision

The complete local launch--rollout--receipt plumbing is executable and its
invariants pass.  Together with the composite cache Lyapunov derivation, this
is sufficient to design a GPU development run.  It is not yet a frozen pilot:
the GPU workload, seeds, critic uncertainty rule, environment stochasticity,
and gates must be committed before any pilot outcomes are read.

The project is no longer blocked by an undefined MARL action.  The online
action is exactly the null action or a directed teammate-policy refresh into a
particular rollout worker's cache.  Gradient packets remain owner-specific and
can return after a random number of later launches.

## Executable CTDE semantics

- There are `n` distinct local actors and one centralized action-value critic.
- Worker `i` always evaluates the current actor `theta_i` for its owner action
  and cached actors `chi_(j->i)` for teammate actions.
- An optional launch action copies one or more complete donor actor payloads
  only into worker `i`; the proposed controller selects at most one.
- A rollout transition is generated under this mixed joint policy.  Replay
  stores downsampled local observations, global state, all actions, shared team
  reward, next state, and termination.
- The critic uses a detached DDPG-style temporal-difference target.
- The actor packet is the deterministic centralized-critic gradient for one
  owner only.  It is cloned at launch and placed in a receipt-time heap.
- Receipt applies fixed-step SGD only to that owner block and increments its
  public version.  A terminal drain accounts for every launched packet.
- Final evaluation uses all current local actors and no cache graph, so
  execution remains decentralized.

The fixed base rollout/gradient traffic is identical for all methods.  The
controlled resource is optional teammate-policy payload bytes.  Actor
transitions count every environment cycle times the number of agents.

## Strong non-oracle schedulers already implemented

The same state machine implements no refresh, age, parameter mismatch,
round-robin, physical-chain, and prefix-feasible complete-burst scheduling.
Age and mismatch never spend a message on an already fresh cache.  All methods
share the same hard prefix allowance

\[
 C_{0:p}\le\lfloor (p+1)\bar c\rfloor,
\]

so a favorable terminal result cannot be purchased by exceeding the declared
budget.  The signed-only controller is retained as an ablation of the exact
cache-energy term.

## CPU validation

The focused suite covers model shapes, deterministic image compression,
distinct recipient/donor caches, exact payload bytes, immutable packet
gradients, delayed owner-only receipt, prefix budgets, predictable age and
mismatch baselines, sparse signed VJP selection, detached TD targets, Polyak
updates, and a real parallel Pistonball rollout:

```text
31 passed in 69.20 s
```

A four-agent, eight-launch, horizon-four composite-controller smoke was run
twice.  After excluding runtime, the complete JSON objects were byte-identical:

```text
SHA-256 92493AE465EB0C08514A95FD6B3F574E0AC59F87CA361C1223B89CEF10B37814
```

It launched 128 actor transitions, received and drained all eight owner
packets, used three of four allowed refreshes, and charged 268,428 optional
policy bytes.  The deterministic action trace was
`1B10467C641ACAA566C39D5AB710E911992C4AE5301ABB8B93F3761143230A60`.

The corresponding learning-only smoke had used zero of four refresh units.
This contrast is the intended anti-absorption mechanism of the exact cache
energy, not a learning-performance claim.

A separate 20-agent smoke used actual Pistonball dimensions, 400 distinct
recipient/donor cache modules, two launches, 40 charged actor transitions, one
received packet, and a complete terminal drain.  It was finite and prefix-
budget feasible.  Its two early owners were outside the launch cone, so the
correct action was null.  Raw SHA-256:

```text
B2237FCE169A5BBFE3EA78F3EDE3C5D87682E6EF5406787E9529FDB86239F6DE
```

All raw development files remain ignored below `tmp/policy_dependency_sync/`.

## Claim boundary

The short development returns are constant because evaluation lasts only
eight cycles; they cannot measure learning.  These smokes do not choose a
budget, Lyapunov weight, seed, or success threshold for a scientific result.

The earlier Pistonball conformal cone used `random_drop=false`,
`random_rotate=false`, and a bounded policy shift.  It cannot be silently
reused as a certificate for the default randomized environment or an
arbitrarily moving neural policy.  A GPU design must either keep that exact
environment/shift scope, preregister a new outcome-free cone calibration, or
label the tube as an architectural prior rather than a theorem certificate.

Similarly, the critic/VJP Taylor coefficient remains an explicit assumption
or estimator input.  The GPU method may use it, but the paper cannot call it a
high-probability neural certificate until a simultaneous Markov-data bound is
proved.

## GPU development gate

The next run should be a development workload, not formal evidence.  It should
answer four questions before preregistration:

1. do the distinct-actor centralized-critic curves learn above initialization;
2. does the composite controller use a nonzero but budget-feasible dynamic
   edge set without collapsing to the mismatch baseline;
3. is there return/sample-efficiency headroom over the strongest age,
   mismatch, complete-burst, and fixed-graph schedulers;
4. is VJP time and memory small enough relative to environment/training time?

Only a positive development phase should be followed by a separately frozen
pilot with new seeds and immutable gates.  Formal seeds remain unauthorized.
