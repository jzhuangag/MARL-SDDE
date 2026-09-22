# T-049A computational Amendment 1

The first post-preregistration execution reached the 3,604.1-second hard
timeout with exit code 124. It generated no row, task-constant file, summary,
or scientific outcome. The runner had created only an empty result directory;
that exact empty path was verified inside the workspace and removed before a
rerun.

The timeout arose because each schedule evaluated the two affine correlation
endpoints independently and repeatedly formed matrix powers inside every
Polyak--Ruppert impulse. Amendment 1 changes only the arithmetic evaluation:

1. the affine intercept and slope are accumulated in one pass;
2. companion responses are generated recursively and their sums are cached;
3. pair contractions sharing a temporal lag are evaluated in a batch.

No task, policy, feature, correlation, delay, overhead, budget, schedule,
probe charge, estimator, comparator, population, threshold, gate, or stop rule
changes. The original configuration SHA-256 remains
`f29c1f6ae56922dab9a5211aec656410fb067d5ceacbb8b7f02acb2fcc83e06f`.

The amended coefficient calculation is tested against the original exact risk
at both correlation endpoints for dimensions 1, 2, and 3; delays 0, 1, and 3;
a nonconstant schedule; and nonidentity positive-definite risk matrices. Every
case agrees within absolute tolerance (2\times10^{-12}). A descriptive
Taxi-v3 benchmark at dimension 16 and horizon 128 changed runtime from about
1.09 seconds to 0.18 seconds without changing the reported coefficients beyond
floating-point summation order.

After the amendment commit, the unchanged deterministic T-049A CPU scan may be
rerun. Sampled trajectories, formal seeds, GPU, and HPC4 remain forbidden.
