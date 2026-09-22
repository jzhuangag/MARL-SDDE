# SMACv2 coupling calibration preregistration

## Scope

`TSP-SMACV2-COUPLING-CAL-001` is an outcome-free integration qualification
for `TSP-MARL-SMACV2-DEV-001`.  It performs no policy update, computes no
return or win rate, and cannot enter a paper as performance evidence.  It is
needed because StarCraft simulation can amplify tiny state differences, so
bitwise-equal trajectories are not the scientific definition of a shared
stochastic regime.

The calibration uses `terran_10_vs_10`, eight rollout workers, 64 blocks of
200 environment ticks, two fresh seeds (`80501`, `80502`) per dependence
regime, the frozen untrained 128-by-128 ReLU MAPPO networks, and the same
residual fingerprint and confidence rule as the development controller.  All
environment interaction is charged descriptively even though no learning is
performed.

## Frozen gates

All four runs must be finite, contain zero policy updates, and expose no
return/win-rate outcome.  Let the regime statistic be the median, over its two
runs, of the within-run median pairwise worker correlation.  The independent
statistic must be at most `0.25`, the shared statistic at least `0.75`, and
their difference at least `0.50`.  Both independent runs must select `q=8`
and both shared runs `q=1` under the already frozen controller rule.

Every gate is mandatory.  Passing authorizes only the existing 24-run
development array.  A failure prohibits that array and cannot be repaired by
lowering a threshold or substituting seeds.  The earlier four-block exact
equality check and its failure remain recorded in
`marl_smacv2_rng_alignment.md`; this calibration does not reinterpret it as a
pass.
