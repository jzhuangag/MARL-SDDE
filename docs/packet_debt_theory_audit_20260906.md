# Packet-debt theorem audit and continuation decision

Date: 2026-09-06

## Decision

The packet-debt construction closes the previously identified causal timing
gap at an abstract stochastic-optimization level.  It is strong enough to
freeze an exact factored-game feasibility experiment after that experiment's
model and gates are separately preregistered.  It is not yet a complete deep
MARL theorem and does not authorize a standard-task learning run or GPU use.

Unlike the earlier launch-time signed-JVP proposal, this construction does not
attempt to predict a packet's realized future return before the packet exists.
The launch action controls a certified quantity that is immediately inserted
into the Lyapunov state.  Chronological event-time telescoping then handles
arbitrary interleavings of bounded-delay launches and receipts.

## Closed obligations

1. **One Lyapunov design:** horizon, version graph, and receipt mass are actions
   of the same objective-plus-packet-debt-plus-resource-queue functional.
2. **No temporal clairvoyance:** launch variables use launch information;
   receipt mass uses receipt information and a sealed independent update half.
3. **Exact graph minimization:** the additive Cauchy certificate makes edge
   inclusion separable; the finite integer-horizon comparison is exact.
4. **Random ordering:** every packet debt is added at birth, updated after each
   intervening policy step, and removed at its own receipt.  Pending boundary
   packets remain as nonnegative terminal energy.
5. **Bias cancellation:** the `5/4` bias coefficient and `1/2` variance
   coefficient yield the proved full-cap descent comparison.
6. **Resource guarantee:** the virtual queue implies a pathwise average
   communication bound and a deterministic cap under bounded scores.
7. **Asymptotic scaling:** for bounded in-flight population and certificates,
   `w=N^(-1/2)` and `V=N` give an `O(N^(-1/2))` optimization/resource remainder
   plus the explicit comparator bias and estimator-error terms.
8. **Executable algebra:** unit tests compare the launch rule against exhaustive
   subset search and the receipt theorem against direct one-dimensional drift
   calculations.

## Assumptions that must remain visible

- The learning objective is a cooperative-team loss or a smooth Markov
  potential.  The theorem is not a convergence result for arbitrary
  general-sum games.
- Each rollout packet has a bounded or deadline-truncated delay certificate.
  Unbounded communication delay receives no finite refresh credit.
- Policy-gradient mean bias admits named Markov, critic, cone, truncation, and
  version components.  A generic neural critic does not supply these constants
  automatically.
- Receipt-time action selection uses a control/update split or a future
  martingale replacement.  Reusing the same random gradient to select and
  execute its step is not covered.
- The number of in-flight packets and cross-policy sensitivities are bounded
  for the stated rate.  Dense interaction is an honest high-cost regime.
- PDSG-CONE-001 gives marginal fixed-reference coverage with policy-shift
  correction, not an anytime simultaneous learning guarantee.

## Still-open obligations

| Obligation | Current status | Required resolution |
|---|---|---|
| receipt score under Markov data | open | expectation-uniform or time-uniform error for centralized critic/control split |
| concrete bias/sensitivity constants | open | exact factored game first; conservative standard-task instantiation later |
| strong-baseline learning headroom | open | outcome-free exact-game oracle gate, then independent CPU controller confirmation |
| owner-sharded systems accounting | open | charge all joint-simulator transitions, policy payloads, control split, and redundant shards |
| controlled-SDDE bridge | open/nonessential | weak finite-horizon approximation before any SDDE guarantee claim |
| standard MARL return evidence | open | only after CPU theorem-facing controller succeeds |

## Stop/go rule

The next CPU stage must first show that, in a nontrivial factored Markov game,
the best dynamic packet-debt action has a meaningful equal-resource gap over
the strongest fixed horizon/static graph envelope after charging the receipt
control split.  If no such gap exists, the controller does not proceed to
Pistonball training even though the cone certificate passed.  If it exists,
the runner and analyzer are frozen before new confirmation seeds, and all
failed gates remain failures.

