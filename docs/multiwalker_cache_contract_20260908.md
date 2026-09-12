## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: outcome-free standard-benchmark contract
- Origin Date: 2026-09-08
- Verification Status: STRUCTURAL CONTRACT PASSED; ORACLE HEADROOM NOT AUTHORIZED YET
- Version Label: multiwalker_cache_contract_v1

# Multiwalker cache/receipt contract

## Scope

This audit establishes only that current PettingZoo Multiwalker can instantiate
the asynchronous policy-cache state machine.  It does not compare learning
returns and does not show that the Lyapunov controller is useful.

The local runtime was repaired by installing the CPython 3.11 Windows wheel
`Box2D==2.3.10`; PettingZoo remains `1.26.1`.  The pinned versions and command
are recorded in `multiwalker_environment_lock_20260908.json` and the project
requirements file.

## Contract

Five distinct actor blocks control five walkers.  At launch for owner `i`, the
owner actor is current and every teammate actor comes from recipient `i`'s
cache.  The declared physical policy-dependency universe is the walker chain:

\[
 \mathcal N_i=\{i-1,i+1\}\cap\{0,\ldots,n-1\}.
\]

The null action sends no optional policy bytes.  A non-null action refreshes
exactly one directed neighbor cache and charges every actor parameter byte.
The actual joint action is produced by the resulting owner-specific mixed
policy profile.  A deterministic owner-version drift then makes outgoing
caches stale, and the packet is inserted into a random delayed receipt heap.

Rewards returned by the environment are deliberately discarded.  The audit
hashes only actions and observations and checks action bounds, finite values,
cache equality, local degree, and receipt ordering.

## Result

Primary and clean reproduction used seeds `95400--95407`, 40 cycles, five
walkers, and maximum delay three.  Both JSON artifacts are byte-identical with
SHA-256
`9D289AB3F523BE0FFFE46C8381B300F8C64DE12E3BE135E5E6A67418085F8476`.
The action/observation trace hash is
`16837140afc00d8ddc0c289dd4893fdbff25f619968eef7f9341c875455deede`.

| Invariant | Result |
|---|---:|
| Launches / receipts | 320 / 320 |
| Non-null one-edge refreshes | 202 |
| Optional actor bytes | 934,048 |
| Maximum local degree | 2 |
| Stale cached actions changed exactly to current action | 192 |
| Action-bound violations | 0 |
| Nonfinite observations | 0 |
| Exact cache-copy failures | 0 |
| Receipt-order failures | 0 |

Two focused tests pass, including complete rerun equality.  Three SWIG wrapper
deprecation warnings are emitted by Box2D but do not change the result.

## Decision

The structural contract passes and authorizes a separate, outcome-free oracle
learning-value design.  It does not authorize a controller pilot, formal
seeds, GPU, or HPC4.  The next gate must derive the continuous-action
profile-factor alignment and demonstrate at least 10% equal-resource oracle
headroom over the declared strong online baseline family before training the
deployable controller.
