# T-050 outcome-free stationary audit plan

This deterministic audit evaluates only theorem coefficients and public task
kernels. It does not read T-049A result rows, sample a trajectory, fit a
threshold, or authorize a controller experiment.

For the three T-049 public Gymnasium tasks it will compute the exact edge
long-run covariance, the PR task constant, and the number of updates needed
for the lifted deterministic contraction to fall below (10^{-3}) and
(10^{-4}) at a half-horizon burn-in. It will also evaluate the T-050
leading fixed-participation phase on the frozen catalogue (q\in\{1,4,16\}),
overheads 8 and 32, and the seven public correlations from T-049A.

The outputs are theorem corollaries and design diagnostics, not formal
empirical evidence. A later sampled experiment requires a separate
preregistration and new identifier.
