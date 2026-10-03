# Causal LSFF conditional-risk confirmation: frozen design

## Purpose

This commit freezes an independent CPU confirmation of the causal resource-debt
mechanism.  It uses 64 new seeds, 92001--92064, disjoint from the 91001--91064
development paths.  The development-selected tradeoff is fixed at `V=4`.
No outcome has been generated at this commit.
The frozen configuration SHA-256 is
`b8e7e3d82d93c48e1e190b3f2da841858e4ac8e730d8b5f587bd29fd20227f3c`.

The Markov risk grid, horizon, strong same-count periodic comparator, oracle,
and exact refresh charges are unchanged from the prior feasibility scan.  The
confirmation produces 20,736 rows.  It evaluates conditional estimation risk,
not an RL return, and cannot authorize GPU work by itself.

## Mandatory gates

1. All rows are finite, and the same-count oracle is never worse than the
   strong periodic comparator.
2. Stationary multiplier-one controls have LSFF/periodic ratio one up to
   `1e-10`.
3. Dynamic geometric LSFF/periodic risk ratio is at most `.90`.
4. LSFF is strictly better than periodic in at least 85% of dynamic rows.
5. Median captured oracle headroom across dynamic rows is at least 70%.
6. Every persistence stratum has geometric ratio at most `.95`.
7. Mean budget utilization is at least 99%, and no hard refresh cap is exceeded.
8. A clean isolated rerun reproduces both artifacts byte for byte.

Any failed gate stops the conditional-risk confirmation.  Passing all gates
authorizes only the separately designed arrival-fresh MPE CPU experiment and
continued theorem closure.  Formal, HPC4, and GPU work remain unauthorized.
