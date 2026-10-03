# T-030 Blackjack Euclidean-certificate nonvacuity audit

## Result

The existing audited affine Markov Theorem 4 is valid in its predictably
decorrelated scope, but its Euclidean-norm constants are not informative
enough to select participation on the registered 280-state Blackjack task.

Exact task constants are:

- dimension: 280;
- minimum stationary probability: `0.00026752750787995`;
- symmetric mean-operator monotonicity:
  `3.4786010953340945e-05`;
- smallest same-time aggregate curvature over the complete registered
  `(q,rho)` catalogue: `0.0030805072039173454`;
- largest usable-update count over every registered q and budget/delay cell:
  2,151.

## Optimistic impossibility for the current bound

Set every term that can only worsen Theorem 4 to its impossible best case:
mixing TV error zero, innovation magnitude and second moment zero, and delay
zero.  Its contraction coefficient reduces to

\[
a(\eta)=1-\eta\mu+2K\eta^2,
\]

whose exact minimizer is `eta=mu/(4K)` and minimum is
`1-mu^2/(8K)`.  Combining the smallest curvature from any action with the
largest update count from any action is additionally optimistic because those
two extrema need not belong to the same feasible action.

Even under this relaxation:

- optimistic eta: `0.0028230749557333536`;
- terminal theorem ratio: `0.9998943876928591`;
- maximum provable reduction: **0.0105612%**.

This is below the 5% nonvacuity gate by roughly a factor of 473.  Adding the
actual innovation residual, finite TV error, or delay can only make the bound
less informative.  Therefore the Euclidean Theorem 4 score is stopped as a
practical Blackjack selector without running another trajectory.

## Consequence

This audit does not invalidate Theorem 4 and is not an impossibility theorem
for the true learning dynamics.  It proves that the current rigorous bound
cannot support the proposed practical-effect claim on this task.  The only
remaining ICML route is a new stationary-weighted MSVE finite-time theorem
whose constants remain explicit under Markov mixing and delay.  Until that
theorem is proved and passes the T-030 analytic gates, new CPU learning runs,
Asterix, GPU, and HPC4 remain unauthorized.
