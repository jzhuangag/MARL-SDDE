## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: post-preregistration theorem-scope audit
- Origin Date: 2026-09-09
- Verification Status: VERIFIED FINITE-HORIZON DYNAMIC-PERFORMANCE REDUCTION; APPROXIMATION QUALITY OPEN
- Version Label: controlled_cache_performance_bound_v1

# When does cache-potential Lyapunov control imply scheduler performance?

## The gap that must not be hidden

The selected-action inequality in the paired-freshness note compares two
actions on the state actually visited by the proposed scheduler. A different
scheduler changes recipient caches, future behavior actions, the Markov state,
and the communication queue. Therefore contextual action regret does **not**
by itself bound the return difference between the two induced trajectories.
Calling that inequality dynamic policy regret would be incorrect.

This distinction is irrelevant to the exact launch/receipt stationarity
telescope, but it is essential for the promised online-performance claim.

## Controlled finite-horizon model

Let `X_t` contain the Markov-game state, current policies, every fixed-universe
cache, in-flight ledger and communication queue. For a launch action `a`, let
`r_t(X_t,a)` be its declared cached-profile utility and let `P_t` be the
nonnegative Lyapunov memory

\[
 P_t(X_t)=H_t(X_t)+\frac{Q_t^2}{2\nu}.
\]

Ignoring an action-independent constant, the composite launch rule

\[
 \max_a\{V r_t(X_t,a)+B_t(a)-Q_t c_t(a)\}
\]

is a greedy policy whose proxy continuation value is

\[
 \widehat V_{t+1}(X_{t+1})=-\frac{P_{t+1}(X_{t+1})}{V}.
\tag{1}
\]

For the cache reset, this identity is exact. For the reflected queue, the
first-order price differs from the exact queue-potential increment by at most

\[
 \zeta_t^Q=\frac{\nu}{2V}
 \sup_a(c_t(a)-\bar c)^2.
\tag{2}
\]

Environment transitions, scheduled actor drift, packet receipts and topology
events between launches must either be included in the transition defining
(1), or contribute their own uniform action-score error. The reset identity
alone covers only the instantaneous cache copy; none of the intervening events
can disappear from the continuation model.

## Finite-horizon performance theorem

Consider the controlled launch MDP induced by the complete event ledger. Let
`V_t^*` be its optimal value-to-go for the same stage utilities and constraints.
Only relative continuation values affect an action. Define the shift-invariant
uniform error over reachable successor states by

\[
 \epsilon_t=
 \inf_{c\in\mathbb R}\sup_x
 \left|\widehat V_{t+1}(x)+c-V_{t+1}^*(x)\right|,
\tag{3}
\]

and the implemented action score differs uniformly from the exact
`r_t+E[widehat V_(t+1)]` score by at most `zeta_t`. If `pi_H` greedily maximizes
that implemented score, then for every initial state

\[
 V_0^*(x)-V_0^{\pi_H}(x)
 \le 2\sum_{t=0}^{N-1}(\epsilon_t+\zeta_t).
\tag{4}
\]

### Proof

Let `a*` maximize the Bellman expression with `V*_(t+1)` and let `a_H` be the
implemented greedy action. Add the optimizing time-dependent constant in (3),
which cancels from every action score. Replacing `V*_(t+1)` by the shifted
`Vhat_(t+1)` costs at most `epsilon_t` for each action; replacing the exact proxy score by the
implemented score costs at most `zeta_t` for each action. The greedy inequality
between `a*` and `a_H` therefore gives one-step loss at most
`2(epsilon_t+zeta_t)`. Add the downstream loss under `a_H` and apply backward
induction. The executable helper
`proxy_greedy_policy_regret_upper` evaluates the right side.

## What this theorem does and does not establish

Equation (4) is a genuine induced-trajectory performance statement. It also
shows the price of avoiding a scheduler Q-network: the negative Lyapunov
memory `-P/V` must approximate the true continuation value. Lyapunov theory is
therefore a design principle, but not a free optimality theorem.

For MW-PF-DEV-001, the privileged `H`-step utility and exact dynamic DP/MILP
allow a direct empirical test of this approximation: does greedy utility plus
the cache reset recover a material fraction of the remaining dynamic oracle
headroom? A failure means `H` is not an adequate continuation proxy on this
model and stops the route. A pass permits a fresh-seed cache-dependent rollout
test, but still does not verify (3) for a neural MARL learner.

For a paper theorem, one of two routes must close:

1. prove (3) from contraction and a task-level relation between cache mismatch
   and continuation value; or
2. estimate a residual correction to `-P/V` and prove a simultaneous
   selected-feedback bound for that correction.

The second route is more general but becomes a controlled-Markov value learner;
its complexity and sample cost must be reported honestly. Neither route may be
replaced by the on-trajectory action-regret inequality.
