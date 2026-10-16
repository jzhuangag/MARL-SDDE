## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: development-experiment Amendment validation
- Origin Date: 2026-09-08
- Verification Status: DEVELOPMENT KILL TEST PASSED; INDEPENDENT CONFIRMATION REQUIRED
- Version Label: multiwalker_budget_oracle_amendment_v1_validation

# Multiwalker exact budget-oracle Amendment validation

## Immutable parent result and amended scope

The original Multiwalker development experiment remains a frozen `6/7`
failure at commit `c9e940a`. Its object labelled `oracle_h8` was a receding
greedy policy, not a prefix-budget optimum. Amendment 1 was frozen at commit
`f8d2af6` before amended outcome generation and changed only this diagnostic:
the environment prefixes, actor snapshots, costs, budgets, seeds, comparator
outcomes, action class, and D4--D7 thresholds were unchanged.

The amended oracle enumerates every null-or-one-edge local cache sequence and
uses a multiple-choice binary program with all 40 prefix-budget constraints to
select one sequence for each of the five owners. The complete-neighbor policy
remains in the strong comparison envelope even though it is outside the
oracle's null-or-one-edge action class.

## CPU execution and provenance

Development seeds `95700--95707` were run as four isolated, disjoint two-seed
CPU chunks and merged only after seed-completeness checks. The local `ust2`
environment was used; no GPU, HPC4, or remote storage was used. The four
chunks completed in approximately 52 minutes of wall time when run in
parallel.

Merged ignored artifacts are under
`tmp/policy_dependency_sync/multiwalker_budget_oracle_amendment_v1/merged`.
Their SHA-256 hashes are:

- `exact_rows.json`: `D214821524866959EEDB63E2C4EC993E7D9A4F17E3F0518DEBA26ADA330B4213`;
- `summary.json`: `11DE2E3483B43B9CE6E6024FEF1B06B9A46A0ECB03794D611374EEC7BE8E598B`.

The original baseline rows hash was rechecked as
`AD79327D849C820C1062C88BB1502A2BB5756064D7EA23EBCFFE6203662B2D42`.

## Frozen gate result

All eight Amendment gates pass on 32 seed--regime cells.

| Gate | Result |
|---|---:|
| A1 complete and finite | pass |
| A2 exact MIP optimum | pass; status optimal and reported gap `0` |
| A3 no-refresh replay | pass; maximum absolute error `8.88e-15` |
| A4 every prefix budget | pass; maximum excess `0` |
| D4 active oracle gain positive | pass; `6.685041` |
| D5 active recovery gap at least 10% | pass; `59.1823%` |
| D6 active positive direction at least 75% | pass; `16/16=100%` |
| D7 median normalized effect at least 0.1% | pass; `1.6119%` |

The active strong-envelope gain over no refresh is `2.728680`; the exact
oracle headroom above that envelope is `3.956361`. Thus the exact dynamic
allocation recovers additional learning value in every active cell rather
than only exploiting one favorable budget.

| Motion scale | Budget rate | Positive cells | Oracle headroom | Median normalized headroom | Oracle gain over no refresh |
|---:|---:|---:|---:|---:|---:|
| `0.01` | `0.25` | `8/8` | `0.403541` | `0.3227%` | `0.653493` |
| `0.01` | `0.50` | `8/8` | `0.524279` | `0.3792%` | `0.827486` |
| `0.04` | `0.25` | `8/8` | `1.694483` | `1.2812%` | `3.039179` |
| `0.04` | `0.50` | `8/8` | `2.261878` | `1.8394%` | `3.645862` |

## Interpretation and next authorization

This is positive development evidence that the standard Multiwalker cache
problem has endogenous, equal-resource dynamic scheduling headroom. It also
locates the v1 directional failure in the greedy diagnostic rather than in a
lack of benchmark value. It is **not** evidence that an observable Lyapunov
controller can recover that value: the exact optimizer uses future branch
values and the same development seeds as v1.

Per the prospective Amendment, this pass authorizes only a separately frozen
independent-seed oracle confirmation. It does not authorize critic fitting,
controller efficacy claims, formal seeds, GPU, or HPC4. The confirmation must
recompute both the strong online envelope and the exact prefix-budget oracle
on untouched seeds. Failure preserves this development result but stops the
Multiwalker efficacy bridge.
