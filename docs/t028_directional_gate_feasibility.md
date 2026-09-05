# T-028 prospective directional-gate feasibility correction

## Scope

T-028 corrects the authorization rule for future experiment identifiers. It
does not edit, reinterpret, or rerun T-020 or T-027. Their frozen failures
remain final.

## Structural proposition

Suppose the registered mechanism risk has the form

```text
R(q,rho) = [rho + (1-rho)/q] / H(q),
```

the environment budget is binding, and its usable horizon is independent of
q. Then `R(q,rho)` is non-increasing in q for every rho in `[0,1]`. The
largest available participation level is therefore a cellwise oracle for
every environment-binding cell. A strong fallback selected from the same
fixed-q catalogue is also the largest q, so no such cell can be strictly
improved by a cellwise fixed-q oracle.

Consequently, when environment-binding cells form fraction `w` of a balanced
design, the maximum possible all-cell strict-improvement fraction is at most
`1-w`. For T-027, `w=1/2`, so the ceiling is 50%, below the frozen 60% gate.
This is a design-feasibility fact, not an experimental outcome.

## Correct future rule

Every future benchmark must partition cells before outcomes using only public
cost/horizon algebra:

- **active cells:** the registered cost geometry permits an internal
  participation optimum or permits the oracle q to change across the declared
  correlation regimes;
- **inactive cells:** monotonicity forces the boundary fallback (for example,
  q=max under a q-independent environment horizon).

The mandatory gates for a new identifier are:

1. at least 5% aggregate oracle improvement over the strong task × budget
   fallback across all registered cells;
2. strict improvement in at least 60% of prospectively declared active cells;
3. inactive cells are all retained and satisfy the prospectively predicted
   fallback tie/boundary behavior;
4. the environment/message budget-direction relation holds in every matched
   pair;
5. at least three fixed-q levels are optimal somewhere in the complete grid;
6. all prior marginal-invariance, mixing, full-cost, taint, and
   reproducibility gates remain mandatory.

The active set must be frozen before any learning trajectory or endpoint is
observed. It cannot be defined by realized oracle improvement.

## Effect on current candidates

T-027 is not rescued: its message population, the natural active subset, has
14/24 strict cells = 58.3333%, still below 60%. FrozenLake remains an exact
mixing/calibration candidate only.

MinAtar Asterix remains the highest-priority external nonlinear candidate,
but it still lacks a prospective mixing/value certificate and therefore does
not yet receive a new experiment identifier or GPU authorization.

