# T-063B collision-gate redesign (outcome-free design note)

This note is a design correction after T-063A.  It does not reopen T-063A,
change its threshold, or authorize a new run.

## Problem diagnosed in T-063A

T-063A used the maximum `rho=0` match rate over 1,536 seed-by-task blocks as
a mandatory gate.  Each block contained 96 fingerprint trials and the frozen
limit was `2%`.  With a finite independent-path collision bound `c_L`, this
maximum statistic has a family-wise multiplicity that grows with the number of
blocks.  A single block with two matches therefore triggered failure even
though the aggregate collision rate was below the certified independent-path
bound.

## Prospective replacement

For every registered `rho=0` seed-task block, retain the 96-bit match count,
but define

\[
 K_{\rm all}=\sum_{s,g}K_{s,g},\qquad
 N_{\rm all}=96\,|\{(s,g)\}|.
\]

Use the public worst-case independent-path probability `c_max` from the frozen
kernel certificate.  The collision gate is an exact one-sided binomial upper
confidence check for the aggregate rate, with family-wise level fixed before
new seeds are generated.  A blockwise maximum is retained only as a
descriptive diagnostic, never as a pass/fail criterion.

For the T-063A primary endpoints, `K_all=65`, `N_all=147456`, and
`c_max=0.0007716049382716049`; the aggregate observed rate is below `c_max`.
This calculation is explanatory only and is not applied retrospectively to
T-063A.

## Safeguards for any new identifier

1. Freeze the exact confidence level, kernel bound, seed registry, runner,
   analyzer, and all efficacy gates in a new independent commit.
2. Use entirely new seed clusters; do not pool them with T-063A for a primary
   claim.
3. Keep the same aggregate/task/delay/breadth/oracle gates unless an
   outcome-free power audit justifies a change.
4. Report T-063A as a formal failure with its original P10 result and retain
   this redesign as a separate prospective validation.
5. Do not start a new run until the new JSON, implementation, tests, and
   hashes pass an outcome-free static audit.
