# EXP-018B frozen analysis plan

The formal seed is the only independent cluster. Projections, q levels, rho
levels, checkpoints, tasks, and mixing profiles are repeated measurements.

The analyzer reconstructs a tensor indexed by seed, stratum, rho, q, and
projection. It estimates projection variances across seeds, averages them over
the 16 fixed projections, and forms 72 q>1/q=1 calibration ratios. The median
and p90 relative errors are the two co-primary endpoints.

The 5,000 bootstrap replicates resample the seed axis globally, preserving all
within-seed common-random-number relationships. Each endpoint uses a one-sided
97.5% percentile upper bound. Both equivalence tolerances must pass.

Adjacent-q directions whose theoretical relative separation is at least 5%
are reported descriptively. Directions below 5% are not counted. Directional
fractions never determine authorization.

Missing, duplicate, nonfinite, unregistered, path-dependent, parameter-changing,
or non-reproducible outputs fail their mandatory gate. No imputation or
outlier removal is allowed.
