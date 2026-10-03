# T-050 stationary theorem audit: results and decision

## Result

The deterministic audit passes its intended mechanism check. It used no
T-049A outcome row, sampled trajectory, seed, GPU, or HPC4 resource. A clean
rerun was byte-identical; the result JSON SHA-256 is
`6457b5880d408e0ea8e5aceee8c81cfc819ce533649eb36c6713c40ad87869c5`.

The exact public-task contraction audit shows that a half-horizon PR burn-in
needs roughly four thousand updates before the deterministic transient is
below (10^{-3}):

| Task | Delay 0 | Delay 4 | Delay 8 |
|---|---:|---:|---:|
| FrozenLake 8x8 | 3,958 | 3,902 | 3,846 |
| CliffWalking | 4,046 | 3,990 | 3,934 |
| Taxi | 4,048 | 3,992 | 3,936 |

The corresponding (10^{-4}) horizons lie between 5,128 and 5,396. These
values follow only from the public lifted spectral radii and the frozen
contraction rule; they were not fitted to T-049A errors.

The exact finite-state long-run covariance is positive semidefinite for every
task. The PR task constants are 1.83152 (FrozenLake), 37,884.03247
(CliffWalking), and 2,555.28865 (Taxi). They scale risk but cancel from the
leading fixed-participation choice.

## Stationary fixed-participation phase

For (q\in\{1,4,16\}) and the seven frozen correlations, the leading
full-information cellwise oracle has substantial value:

| Message overhead | Strong fixed q | Oracle geometric gain | Strict cells | Oracle support over increasing rho |
|---:|---:|---:|---:|---|
| 8 | 4 | 16.5035% | 5/7 | 16, 16, 4, 4, 1, 1, 1 |
| 32 | 16 | 13.1690% | 4/7 | 16, 16, 16, 4, 4, 1, 1 |

This is a theorem coefficient, not a sampled performance claim. It is
nevertheless enough to identify the failure mechanism: T-049A evaluated
96/192-update transients, while the public contraction threshold is about
4,000 updates. At stationarity the fixed-action correlation phase is broad;
the old time-varying schedule phase remains absent.

## Decision

T-050 authorizes one new outcome-free design stage, T-051, for a low-cost
independent-probe then fixed-(q) controller. T-051 must:

1. retain the unchanged Gymnasium task marginals and the T-049
   trajectory-switch coupling;
2. choose its learning horizon from the (10^{-3}) contraction rule before
   sampled outcomes;
3. charge every probe message, environment transition, and delay;
4. prove its observable correlation statistic and error bound;
5. demonstrate at least 5% certified full-cost static value before sampling;
6. use a new identifier and seeds; it may not revive EXP-021A.

T-051 remains a local CPU/theory stage. GPU and HPC4 are not authorized.
