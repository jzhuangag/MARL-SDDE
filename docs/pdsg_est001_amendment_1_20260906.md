# PDSG-EST-001 Amendment 1: valid finite-action uniform error

Date: 2026-09-06

Status: frozen before runner implementation and before any sampled output.

The original manifest remains immutable with SHA-256
`953EEDAE61D00B3E637573F83452273538187CFB4FAFF85CDE8D26070DD93146`.
A static formula audit found that its actionwise-sum error rule grows linearly
with the dependency degree even for jointly Gaussian errors.  The rule is
valid but discards the shared finite-action structure that the registered
Gaussian model provides.

Amendment 1 replaces only that aggregation with the proved Gaussian
expected-maximum norm bound in Equation (11) of
`policy_dependency_alignment_estimator_20260906.md`.  Dependence among action
errors remains allowed; the degree dependence becomes `sqrt(log(Delta+1))`.
The sampled noise variances, block lengths, Markov correlations, degrees,
displacements, seeds, and numerical gate thresholds are unchanged.

The favorable population is made explicit as the exact-sparse, locally linear,
sign-separated class (`M=0`, sparse tail zero, true margin at least 0.04).  The
high-correlation, short-block version of that same class is the adverse
population.  Nonzero Taylor and sparse-tail cells remain required descriptive
robustness controls; they are not silently treated as exact-sparse theory.

The scenario geometry is now fully specified: the current gradient has unit
norm, the no-refresh gradient has cosine 0.3, and the candidate cosines are
listed by degree in the amended manifest.  Candidate labels and orthogonal
directions are permuted by the already frozen scenario seed.  The exactly
observed reset benefit equals the queue price for each edge, so the audit tests
signed learning-score estimation rather than a trivial constant edge bonus.

No trajectory, noisy score, ranking accuracy, or gate outcome existed when
this amendment was written.  From this commit onward, the v2 manifest is
immutable.  The original E1--E9 stopping logic remains: failure of E1--E8
forbids reproduction and any successor learning pilot.
