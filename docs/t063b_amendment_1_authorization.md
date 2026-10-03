# T-063B Amendment 1 authorization

The outcome-free static audit for Amendment 1 passed before this authorization
commit: the amended configuration hash, base provenance, targeted tests, full
nonlinear test suite, and artifact-cleanliness checks were all valid.

This separate commit authorizes one serial local-CPU formal execution under the
amended specification. It does not authorize GPU, HPC4, project storage, any
change to the frozen seeds or gates, or any retrospective reclassification of
T-063A.

The amended replay gate uses recursive absolute and relative tolerance
`1e-12` for numeric summary values while retaining byte-exact equality for all
primary/reproduction artifacts. The resulting configuration hash is recorded
in the JSON preregistration and must be checked by the runner before execution.
