# T-061A reward-free MinAtar controller CPU pilot preregistration

## Confirmatory question

T-061A prospectively tests whether an observable, reward-free correlation
probe can transfer the closed-form participation phase to a standard
high-dimensional reinforcement-learning benchmark after paying every probe,
message, actor-transition, and delay cost.

T-060A remains a failed experiment.  Its noisy cellwise empirical selector is
not reused.  Its frozen reference moments and selection-half strong fixed-q
table are design inputs, while all T-061A learning and probe streams use 32
new master seeds.  T-060A's post-result theory-rule audit supplies power and
design motivation only, not evidence for T-061A.

## Reward-free controller

Each of 96 independent probe blocks generates one common and two private
four-transition MinAtar paths.  Two probe actors select the common path with
probability `sqrt(rho)` and their private paths otherwise.  A transmitted
fingerprint hashes the state, previous executed action, requested action,
reward, and terminal tuple at every transition.  Because the policy requests
one of six actions uniformly, two independent full fingerprints can match
with probability at most `6^-4=0.000771605`.

The controller forms `rho_hat=K/96` from the match count and chooses q in
`{1,4,16}` minimizing

`(overhead+q) * (rho_hat+(1-rho_hat)/q)`.

No reward or TD loss comparison enters this participation decision.  Online
selection is one integer counter plus a three-action scan.  Learning uses the
same frozen nonlinear encoder and 33-dimensional delayed regularized TD head
as T-060A.

## Full costs and comparators

The total budgets remain `(overhead+16)*8192` messages and `16*8192` actor
transitions.  Before learning, the controller pays `96*(overhead+2)` message
units and `96*2*4` actor transitions.  Its usable updates are computed from
both remaining budgets and then reduced by delay.  The task-by-overhead strong
fixed-q comparator pays no probe and receives the entire total budgets.  A
no-probe full-budget true-rho phase rule is descriptive and measures probe
classification/opportunity cost; it cannot replace the strong primary
baseline.

## Frozen gates

P1--P11 are jointly mandatory:

1. exactly 2,688 unique registered endpoints;
2. finite positive risks and exact full message/environment/delay accounting;
3. aggregate controller/strong geometric ratio at most 0.95;
4. taskwise ratio at most 0.98 for each of Asterix, Breakout, and Seaquest;
5. ratio at most 0.97 at each delay;
6. strict controller improvement in at least 60% of cells;
7. aggregate controller/full-budget-true-rho ratio at most 1.15;
8. median participation nonincreasing with rho on at least 75% of
   task-overhead-delay paths;
9. fingerprint standardized residual RMSE at most 1.5;
10. every seed-level rho-zero probe match rate at most 0.02;
11. exact new-seed coverage with no historical seed reuse.

A clean rerun must reproduce endpoints, cells, and summary byte for byte and
must replay the full summary from endpoints.  This post-run condition is
required in addition to P1--P11.

## Stop rule and compute

Any failed gate prevents formal registration and nonlinear controller claims;
tasks, seeds, probe, thresholds, budgets, or endpoints cannot be changed to
rescue T-061A.  Passing authorizes only an independent formal power audit and
formal preregistration.  The frozen estimate is about 20 million generated
transitions and 1.5 GiB peak memory, so the pilot runs on the local CPU.  GPU
and HPC4 remain unauthorized.
