# AC-9 uniform matching audit

## Unrestricted claim: impossible

The original phrase “uniform constant/log matching” cannot include the whole
controlled-belief instance class. Consider a fixed separated identification
pair and move continuously toward an oracle action transition. The terminal
risk expansion from (5) gives baseline-to-oracle gain
\(g/s+O(s^{-2})\), where the coefficient gap \(g\downarrow0\). Any reliable
probe requires a nonzero number \(n_\delta\) of observations and therefore
incurs opportunity loss at least of order \(n_\delta/s^2\). Adaptation cannot
be certified worthwhile before

\[
 s\gtrsim \frac{n_\delta}{g}.                               \tag{8}
\]

The identification lower threshold remains bounded along this sequence,
while (8) diverges. Hence \(B_S/B_N\to\infty\). No universal constant or
logarithmic factor independent of the instance can compare the safe
adaptation threshold with the identification threshold.

The same obstruction appears when the safe gain tends to zero, the catalogue
becomes ill-conditioned, or the admissible safety slack vanishes. Separately,
AC-8 rules out uniformity as \(\lambda\uparrow1\), and
\(|\theta_1-\theta_0|\downarrow0\) makes identification information vanish.
These are theorem-level degeneracies, not tuning failures.

## Repaired AC-9

AC-9 is split into three statements:

1. **Unrestricted uniform matching:** closed negatively by (8), reinforced
   by the AC-8 boundary theorem.
2. **Compact separated finite-budget threshold sandwich:** proved under the
   explicit \(\Delta_{\min},\gamma,g_{\min}\), finite-catalogue, positive-cost,
   bounded-delay assumptions in `adaptation_threshold_sandwich.md`.
3. **Matching a history-dependent information-directed controller to the
   entire controlled-Kalman-belief occupation optimum:** still open.

Thus “AC-9 proved” is allowed only when followed by “finite-budget threshold
sandwich on the declared separated class.” It must not be shortened to a
global adaptive-optimality claim.

## What remains distinct from existing controlled sensing

Generic causal-control factorization, cancellation of common action kernels,
binary change of measure, and information-per-control-cost occupation
programs are inherited machinery. The remaining contribution candidate is
the coupled *learning-value* threshold:

- \(q\) changes both common-factor information and communication cost;
- \(b\) changes both decorrelation and environment cost;
- correlation controls the attainable post-identification speedup;
- delay consumes usable learning updates but not observation information;
- a wrong decision changes the downstream optimizer and risk;
- a baseline-relative safety deficit must be paid together with
  identification error.

These coupled quantities create (3)--(8). They are not supplied by a pure
controlled-hypothesis-testing or pure best-arm-identification theorem.
