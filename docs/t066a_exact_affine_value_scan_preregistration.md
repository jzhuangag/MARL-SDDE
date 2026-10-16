# T-066A preregistration: exact delayed-affine value scan

T-066A is an outcome-free analytic falsification gate between the positive
T-065A surrogate-score pilot and any sampled delayed-TD experiment.  It asks a
harder question: after every sensor transition and message actually reduces
the learning horizon, is there still meaningful finite-risk value in adapting
`q` and `eta` jointly?

The scan propagates exact first and second moments of the delayed affine
recursion.  It evaluates terminal mean-square parameter error, not the online
controller's own drift score.  There are no random seeds, sampled outcomes,
confidence intervals, or benchmark claims.

The frozen design contains three public two-dimensional drift matrices, three
delays, four correlations, three noise scales, three initial-error energies,
two dual-budget regimes, six participation levels, and seven gains.  This is
648 cells and 27,216 exact action rows.  Every gain lies inside the T-065
common-certificate interval `[0.005,0.02]`.

The primary static gates require at least 10% aggregate oracle improvement
over strong task-by-budget fixed pairs, strict improvement in at least 60% of
cells, and strict separation from both one-dimensional restrictions in at
least 30% of cells.  Any failure stops a sampled T-066 pilot.  Exact details
and all gates are frozen in
`docs/t066a_exact_affine_value_scan_preregistration.json`.
